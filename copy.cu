#include <cuda.h>
#include <stdlib.h>
#include "util.h"

using namespace cute;

template <typename T, typename G2SCopy, typename S2RCopy, typename SmemLayout, int M, int N>
__global__ void copy_global_shm_register(const T *Aptr)
{
    int idx = threadIdx.x;
    extern __shared__ T shm_data[];
    T *Ashm = shm_data;

    auto gA = make_tensor(make_gmem_ptr(Aptr), make_shape(Int<M>{}, Int<N>{}), make_stride(Int<N>{}, Int<1>{}));

    auto sA = make_tensor(make_smem_ptr(Ashm), SmemLayout{});

    auto rA = make_tensor_like(gA);

    G2SCopy g2s_tiled_copy;
    auto g2s_thr_copy = g2s_tiled_copy.get_slice(idx);
    auto tAgA = g2s_thr_copy.partition_S(gA);
    auto tAsA = g2s_thr_copy.partition_D(sA);
    cute::copy(g2s_tiled_copy, tAgA, tAsA);

    S2RCopy s2r_tiled_copy;
    auto s2r_thr_copy = s2r_tiled_copy.get_slice(idx);
    // error: In CopyAtom, src layout doesn't vectorize into registers. This src layout is incompatible with this tiled copy.
    // auto stAsA = s2r_thr_copy.partition_S(sA);
    auto stAsA = s2r_thr_copy.retile_S(tAsA);
    auto tArA = s2r_thr_copy.partition_D(rA);
    cute::copy(s2r_tiled_copy, stAsA, tArA);

    if (idx == 0)
    {
        // ((_8,_1),_4,_4)
        // (CPY, CPY_M, CPY_N)
        // 其中CPY由copy_op決定，這裡對2個OP都是128bit=16bytes=8half, 確定了copy的基本操作單位。
        // CPY_M = M / (ThrLayout_M * ValLayout_M), CPY_N = N / (ThrLayout_N * ValLayout_N), CPY_M、CPY_N和TiledCopy共同確定了拷貝的元素數量。

        PRINT("tAgA", tAgA.shape());
        PRINT("tAsA", tAsA.shape());
        PRINT("stAsA", stAsA.shape());
        PRINT("tArA", tArA.shape());
    }
}


int main()
{
    using T = cute::half_t;
    int device;
    cudaError_t err = cudaGetDevice(&device);
    if (err != cudaSuccess)
    {
        fprintf(stderr, "cudaGetDevice returned error %d (%s)\n", err, cudaGetErrorString(err));
        return -1;
    }

    int sharedMemPerBlock;
    err = cudaDeviceGetAttribute(&sharedMemPerBlock, cudaDevAttrMaxSharedMemoryPerBlock, device);
    if (err != cudaSuccess)
    {
        fprintf(stderr, "cudaDeviceGetAttribute returned error %d (%s)\n", err, cudaGetErrorString(err));
        return -1;
    }

    printf("Max shared memory per block for device %d is %d bytes\n", device, sharedMemPerBlock);

    // prefer more shm and less L1 cache
    err = cudaDeviceSetCacheConfig(cudaFuncCachePreferShared);
    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to set cache config (%s)\n", cudaGetErrorString(err));
        exit(-1);
    }

    // constexpr int M = 128;
    // constexpr int N = 32;
    constexpr int M = 128;
    constexpr int N = 128;

    cudaEvent_t start, end;
    float elapsedTime;

    cudaEventCreate(&start);
    cudaEventCreate(&end);

    // global memory to shared memory copy
    // 利用cp.async完成global memory到shared memory的异步拷贝。
    // 每个线程完成128bit=16bytes的数据拷贝
    using g2s_copy_op = SM80_CP_ASYNC_CACHEGLOBAL<cute::uint128_t>;
    using g2s_copy_traits = Copy_Traits<g2s_copy_op>;
    using g2s_copy_atom = Copy_Atom<g2s_copy_traits, T>;
    /** Produce a TiledCopy from logical thread and values layouts.
     * The thread and value layouts map coordinates to thr_idx and val_idx.
     *    The product of these layouts is taken to produce the TV layout and the Tiler.
     * Useful when threads and values need very specific mappings onto coordinates
     *    in the target tensors.
     *
     *
     * make_tiled_copy(Copy_Atom<Args...> const& copy_atom,
                ThrLayout          const& thr_layout = {},     // (m,n) -> thr_idx
                ValLayout          const& val_layout = {})     // (m,n) -> val_idx
     */
    // (32,4) threads layout, every thread handle 8 elements(8half=16bytes=128bit)
    using G2SCopy =
        decltype(make_tiled_copy(g2s_copy_atom{},
                                 make_layout(make_shape(Int<32>{}, Int<4>{}),
                                             make_stride(Int<4>{}, Int<1>{})),
                                 make_layout(make_shape(Int<1>{}, Int<8>{}))));

    // logical layout (8,32) to physical layout (8,8,8)
    using SmemLayoutAtom = decltype(composition(
        Swizzle<3, 3, 3>{},
        make_layout(make_shape(Int<8>{}, Int<32>{}),
                    make_stride(Int<32>{}, Int<1>{}))));
    using SmemLayout = decltype(tile_to_shape(SmemLayoutAtom{},
                                              make_shape(Int<M>{}, Int<N>{})));

    static constexpr int shm_size = cute::cosize(SmemLayout{}) * sizeof(T);

    // shared memory to register copy
    /*
    ldmatrix可以实现warp level共享内存到寄存器的数据搬运。
    ldmatrix由于是单线程提供16Byte的数据地址，warp内所有线程可以提供512Byte的数据到寄存器的加载，单指令实现16x16 float16矩阵的加载，减少指令数。
    */
    using s2r_copy_op = SM75_U32x4_LDSM_N;
    using s2r_copy_traits = Copy_Traits<s2r_copy_op>;
    using s2r_copy_atom = Copy_Atom<s2r_copy_traits, T>;
    using S2RCopy =
        decltype(make_tiled_copy(s2r_copy_atom{},
                                 make_layout(make_shape(Int<32>{}, Int<4>{}),
                                             make_stride(Int<4>{}, Int<1>{})),
                                 make_layout(make_shape(Int<1>{}, Int<8>{}))));

    T *Aptr;
    cudaMalloc(&Aptr, sizeof(T) * M * N);
    dim3 block(128);
    cudaEventRecord(start);
    int count = 100;
    for (int i = 0; i < count; ++i)
    {
        copy_global_shm_register<T, G2SCopy, S2RCopy, SmemLayout, M, N><<<1, block, shm_size>>>(Aptr);
    }
    cudaEventRecord(end);
    cudaEventSynchronize(end);
    err = cudaGetLastError();
    if (err == cudaSuccess)
    {
        printf("err = %d, str = %s\n", err, cudaGetErrorString(err));
    }
    else
    {
        printf("err = %d, str = %s\n", err, cudaGetErrorString(err));
    }
    cudaEventElapsedTime(&elapsedTime, start, end);
    std::cout << "copy_global_shm_register took " << elapsedTime / count << "ms." << std::endl;
}