#include <cuda.h>
#include <stdlib.h>
#include "util.h"

// #define PRINT_INFO
using namespace cute;

template <typename T>
void gen_rand_data(T *data, int n);

template <typename T, int kTileM, int kTileN, int kTileK, typename TiledMMA>
__global__ void gemm_simple(T *Cptr, const T *Aptr, const T *Bptr, int m, int n, int k)
{

  Tensor A = make_tensor(make_gmem_ptr(Aptr), make_shape(m, k), make_stride(k, Int<1>{}));
  Tensor B = make_tensor(make_gmem_ptr(Bptr), make_shape(n, k), make_stride(k, Int<1>{}));
  Tensor C = make_tensor(make_gmem_ptr(Cptr), make_shape(m, n), make_stride(n, Int<1>{}));

  int ix = blockIdx.x;
  int iy = blockIdx.y;

  //  gA(kTileM, kTileK, num_tile_k)
  //  gB(kTileN, kTileK, num_tile_k)
  //  gC(kTileM, kTileN)
  Tensor gA = local_tile(A, make_tile(Int<kTileM>{}, Int<kTileK>{}), make_coord(iy, _));
  Tensor gB = local_tile(B, make_tile(Int<kTileN>{}, Int<kTileK>{}), make_coord(ix, _));
  Tensor gC = local_tile(C, make_tile(Int<kTileM>{}, Int<kTileN>{}), make_coord(iy, ix));

  TiledMMA tiled_mma;
  auto thr_mma = tiled_mma.get_slice(threadIdx.x);
  // MMA_M = M / (mma_op_m * thr_layout_m)
  // MMA_N = N / (mma_op_n * thr_layout_n)
  // MMA_K = K / (mma_op_k * thr_layout_k)
  auto tAgA = thr_mma.partition_A(gA); // (MMA, MMA_M, MMA_K, num_tile_k)
  auto tBgB = thr_mma.partition_B(gB); // (MMA, MMA_N, MMA_K, num_tile_k)
  auto tCgC = thr_mma.partition_C(gC); // (MMA, MMA_M, MMA_N)

  // register tensor, shape is different with global memory tensor
  auto tArA = thr_mma.partition_fragment_A(gA(_, _, 0)); // (MMA, MMA_M, MMA_K)
  auto tBrB = thr_mma.partition_fragment_B(gB(_, _, 0)); // (MMA, MMA_N, MMA_K)
  auto tCrC = thr_mma.partition_fragment_C(gC(_, _));    // (MMA, MMA_M, MMA_N)
  clear(tCrC);

#ifdef PRINT_INFO
/*
    gA : (_128,_32,32)
    tAgA : ((_2,_2,_2),_4,_2,32)
    tArA : ((_2,_2,_2),_4,_2)
    gB : (_128,_32,32)
    tBgB : ((_2,_2),_8,_2,32)
    tBrB : ((_2,_2),_8,_2)
    gC : (_128,_128)
    tCgC : ((_2,_2),_4,_8)
    tCrC : ((_2,_2),_4,_8)
*/
  if (threadIdx.x == 0 && blockIdx.x == 0 && blockIdx.y == 0)
  {

    PRINT("gA", gA.shape())     
    PRINT("tAgA", tAgA.shape()) 
    PRINT("tArA", tArA.shape()) 

    PRINT("gB", gB.shape())     
    PRINT("tBgB", tBgB.shape()) 
    PRINT("tBrB", tBrB.shape()) 

    PRINT("gC", gC.shape())     
    PRINT("tCgC", tCgC.shape()) 
    PRINT("tCrC", tCrC.shape()) 
  }
#endif

  int num_tile_k = size<2>(gA);
#pragma unroll 1
  for (int itile = 0; itile < num_tile_k; ++itile)
  {
    // global memory to register
    // just use cute::copy, not tiled
    copy(tAgA(_, _, _, itile), tArA);
    copy(tBgB(_, _, _, itile), tBrB);

    // warp level, use  tiled_mma
    gemm(tiled_mma, tCrC, tArA, tBrB, tCrC);
  }
  // register to global memory
  copy(tCrC, tCgC);
}

int main()
{
  srand(1000);

  using T = cute::half_t;
  cudaEvent_t start, end;
  float elapsedTime;
  cudaEventCreate(&start);
  cudaEventCreate(&end);

  T *Cptr;
  T *Aptr;
  T *Bptr;

  int m = 1024*64;
  int n = 128;
  int k = 1024;

  cudaMalloc(&Cptr, sizeof(T) * m * n);
  cudaMalloc(&Aptr, sizeof(T) * m * k);
  cudaMalloc(&Bptr, sizeof(T) * k * n);

  T *Aptr_host;
  T *Bptr_host;
  Aptr_host = (T *)malloc(sizeof(T) * m * k);
  Bptr_host = (T *)malloc(sizeof(T) * n * k);
  gen_rand_data(Aptr_host, m * k);
  gen_rand_data(Bptr_host, n * k);

  cudaMemcpy(Aptr, Aptr_host, sizeof(T) * m * k, cudaMemcpyHostToDevice);
  cudaMemcpy(Bptr, Bptr_host, sizeof(T) * n * k, cudaMemcpyHostToDevice);

  // M=16,N=8,K=16
  using mma_op = SM80_16x8x16_F16F16F16F16_TN;
  using mma_traits = MMA_Traits<mma_op>;
  using mma_atom = MMA_Atom<mma_traits>;
  using MMA = decltype(make_tiled_mma(mma_atom{},
                                      make_layout(Shape<_2, _2, _1>{}),   // the best shape for perf, use 2x2=4 warps = 128 threads
                                      make_layout(Shape<_1, _1, _1>{}))); // influence little to perf

  constexpr int kTileM = 128;
  constexpr int kTileN = 128;
  constexpr int kTileK = 32;

  // each thread block handle with (kTileM, kTileN) output
  dim3 grid(n / kTileN, m / kTileM);
  dim3 block(size(MMA{}));

  int count = 100;
  cudaEventRecord(start);
  for (int i = 0; i < count; ++i)
  {
    gemm_simple<T, kTileM, kTileN, kTileK, MMA><<<grid, block>>>(Cptr, Aptr, Bptr, m, n, k);
  }
  auto err = cudaGetLastError();
  printf("err = %d, str = %s\n", err, cudaGetErrorString(err));
  cudaEventRecord(end);
  cudaEventSynchronize(end);
  cudaEventElapsedTime(&elapsedTime, start, end);
  std::cout << "gemm-simple took " << elapsedTime / count << "ms." << std::endl;
}

template <typename T>
void gen_rand_data(T *data, int n)
{
  for (int i = 0; i < n; ++i)
  {
    float v = (rand() % 200 - 100) * 0.01;
    data[i] = v;
  }
}
