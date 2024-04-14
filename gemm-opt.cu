#include <cuda.h>
#include <stdarg.h>
#include <stdio.h>
#include "util.h"
#include "detail/cublaslt-gemm.h"
#include "detail/data.h"

// #define PRINT_INFO
using namespace cute;

namespace config
{
  using namespace cute;

  template <typename T_, int kTileM_ = 128, int kTileN_ = 128, int kTileK_ = 32>
  struct GemmConfigV1
  {
    using T = T_;

    static constexpr int kTileM = kTileM_;
    static constexpr int kTileN = kTileN_;
    static constexpr int kTileK = kTileK_;

    // shared memory layout
    using SmemLayoutAtom = decltype(composition(
        Swizzle<3, 3, 3>{},
        make_layout(make_shape(Int<8>{}, Int<kTileK>{}),
                    make_stride(Int<kTileK>{}, Int<1>{}))));
    using SmemLayoutA = decltype(tile_to_shape(SmemLayoutAtom{},
                                               make_shape(Int<kTileM>{}, Int<kTileK>{})));
    using SmemLayoutB = decltype(tile_to_shape(SmemLayoutAtom{},
                                               make_shape(Int<kTileN>{}, Int<kTileK>{})));

    // mma
    using mma_op = SM80_16x8x16_F16F16F16F16_TN;
    using mma_traits = MMA_Traits<mma_op>;
    using mma_atom = MMA_Atom<mma_traits>;
    using MMA = decltype(make_tiled_mma(mma_atom{},
                                        make_layout(Shape<_2, _2, _1>{}),
                                        make_layout(Shape<_1, _2, _1>{})));

    // copy from global memory to shared memory
    using g2s_copy_op = SM80_CP_ASYNC_CACHEGLOBAL<cute::uint128_t>;
    using g2s_copy_traits = Copy_Traits<g2s_copy_op>;
    using g2s_copy_atom = Copy_Atom<g2s_copy_traits, T>;
    using G2SCopyA =
        decltype(make_tiled_copy(g2s_copy_atom{},
                                 make_layout(make_shape(Int<32>{}, Int<4>{}),
                                             make_stride(Int<4>{}, Int<1>{})),
                                 make_layout(make_shape(Int<1>{}, Int<8>{}))));
    using G2SCopyB = G2SCopyA;

    // copy from shared memory to register
    // use mma tiled ,so no tiled here
    using s2r_copy_op = SM75_U32x4_LDSM_N;
    using s2r_copy_traits = Copy_Traits<s2r_copy_op>;
    using s2r_copy_atom = Copy_Atom<s2r_copy_traits, T>;
    using S2RCopyAtomA = s2r_copy_atom;
    using S2RCopyAtomB = s2r_copy_atom;

    // C_shm is shared with A_shm and B_shm
    static constexpr int shm_size_AB =
        cute::cosize(SmemLayoutA{}) + cute::cosize(SmemLayoutB{});
    static constexpr int kShmSize =
        shm_size_AB * sizeof(T);
  };

  template <typename T_, int kTileM_ = 128, int kTileN_ = 128, int kTileK_ = 32,
            int kStage_ = 3, int kSmemLayoutCBatch_ = 2>
  struct GemmConfig
  {
    using T = T_;

    static constexpr int kTileM = kTileM_;
    static constexpr int kTileN = kTileN_;
    static constexpr int kTileK = kTileK_;
    static constexpr int kStage = kStage_;
    static constexpr int kSmemLayoutCBatch = kSmemLayoutCBatch_;

    // shared memory layout
    using SmemLayoutAtom = decltype(composition(
        Swizzle<3, 3, 3>{},
        make_layout(make_shape(Int<8>{}, Int<kTileK>{}),
                    make_stride(Int<kTileK>{}, Int<1>{}))));
    using SmemLayoutA = decltype(tile_to_shape(SmemLayoutAtom{},
                                               make_shape(Int<kTileM>{}, Int<kTileK>{}, Int<kStage>{})));
    using SmemLayoutB = decltype(tile_to_shape(SmemLayoutAtom{},
                                               make_shape(Int<kTileN>{}, Int<kTileK>{}, Int<kStage>{})));

    // mma
    using mma_op = SM80_16x8x16_F16F16F16F16_TN;
    using mma_traits = MMA_Traits<mma_op>;
    using mma_atom = MMA_Atom<mma_traits>;
    using MMA = decltype(make_tiled_mma(mma_atom{},
                                        make_layout(Shape<_2, _2, _1>{}),
                                        make_layout(Shape<_1, _2, _1>{}))); // should obey TiledNumVal{} % AtomNumVal{} == Int<0>{}

    // copy from global memory to shared memory
    using g2s_copy_op = SM80_CP_ASYNC_CACHEGLOBAL<cute::uint128_t>;
    using g2s_copy_traits = Copy_Traits<g2s_copy_op>;
    using g2s_copy_atom = Copy_Atom<g2s_copy_traits, T>;
    using G2SCopyA =
        decltype(make_tiled_copy(g2s_copy_atom{},
                                 make_layout(make_shape(Int<32>{}, Int<4>{}),
                                             make_stride(Int<4>{}, Int<1>{})),
                                 make_layout(make_shape(Int<1>{}, Int<8>{}))));
    using G2SCopyB = G2SCopyA;

    // copy from shared memory to register
    // use mma tiled ,so no tiled here
    using s2r_copy_op = SM75_U32x4_LDSM_N;
    using s2r_copy_traits = Copy_Traits<s2r_copy_op>;
    using s2r_copy_atom = Copy_Atom<s2r_copy_traits, T>;
    using S2RCopyAtomA = s2r_copy_atom;
    using S2RCopyAtomB = s2r_copy_atom;

    // register to global via shared memory
    using MNK = typename MMA::TiledShape_MNK; // Shape<_16,_8,_16>
    using SmemLayoutAtomC = decltype(composition(
        Swizzle<2, 3, 3>{}, make_layout(make_shape(get<0>(MNK{}), get<1>(MNK{})),
                                        make_stride(get<1>(MNK{}), Int<1>{}))));
    using SmemLayoutC = decltype(tile_to_shape(
        SmemLayoutAtomC{},
        make_shape(get<0>(MNK{}), get<1>(MNK{}), Int<kSmemLayoutCBatch>{})));

    static_assert(size<0>(SmemLayoutA{}) * size<1>(SmemLayoutA{}) >=
                      size(SmemLayoutC{}),
                  "C shared memory request is large than A's one pipe");
    // copy from register to shared memory
    using R2SCopyAtomC = Copy_Atom<UniversalCopy<int>, T>;
    // copy from shared memory to global memory
    using S2GCopyAtomC = Copy_Atom<UniversalCopy<cute::uint128_t>, T>;
    using S2GCopyC =
        decltype(make_tiled_copy(S2GCopyAtomC{},
                                 make_layout(make_shape(Int<32>{}, Int<4>{}),
                                             make_stride(Int<4>{}, Int<1>{})),
                                 make_layout(make_shape(Int<1>{}, Int<8>{}))));

    // C_shm is shared with A_shm and B_shm
    static constexpr int shm_size_AB =
        cute::cosize(SmemLayoutA{}) + cute::cosize(SmemLayoutB{});
    static constexpr int shm_size_C = cute::cosize(SmemLayoutC{});
    static constexpr int kShmSize =
        cute::max(shm_size_AB, shm_size_C) * sizeof(T);
  };

} // namespace config

// apply shm
template <typename Config>
__global__ void
gemm_opt_shm(void *Dptr, const void *Aptr, const void *Bptr, int m, int n,
             int k)
{
  using T = typename Config::T;
  using SmemLayoutA = typename Config::SmemLayoutA;
  using SmemLayoutB = typename Config::SmemLayoutB;
  using TiledMMA = typename Config::MMA;

  using S2RCopyAtomA = typename Config::S2RCopyAtomA;
  using S2RCopyAtomB = typename Config::S2RCopyAtomB;
  using G2SCopyA = typename Config::G2SCopyA;
  using G2SCopyB = typename Config::G2SCopyB;

  constexpr int kTileM = Config::kTileM;
  constexpr int kTileN = Config::kTileN;
  constexpr int kTileK = Config::kTileK;

  extern __shared__ T shm_data[];

  T *Ashm = shm_data;
  T *Bshm = shm_data + cute::cosize(SmemLayoutA{});

  int idx = threadIdx.x;
  int ix = blockIdx.x;
  int iy = blockIdx.y;

  Tensor A = make_tensor(make_gmem_ptr((T *)Aptr), make_shape(m, k),
                         make_stride(k, Int<1>{})); // (M, K)
  Tensor B = make_tensor(make_gmem_ptr((T *)Bptr), make_shape(n, k),
                         make_stride(k, Int<1>{})); // (N, K)
  Tensor D = make_tensor(make_gmem_ptr((T *)Dptr), make_shape(m, n),
                         make_stride(n, Int<1>{})); // (M, N)
  // global memory
  Tensor gA = local_tile(A, make_tile(Int<kTileM>{}, Int<kTileK>{}),
                         make_coord(iy, _)); // (kTileM, kTileK, k)
  Tensor gB = local_tile(B, make_tile(Int<kTileN>{}, Int<kTileK>{}),
                         make_coord(ix, _)); // (kTileN, kTileK, k)
  Tensor gD = local_tile(D, make_tile(Int<kTileM>{}, Int<kTileN>{}),
                         make_coord(iy, ix)); // (kTileM, kTileN)

  // shared memory
  auto sA = make_tensor(make_smem_ptr(Ashm),
                        SmemLayoutA{}); // (kTileM, kTileK)
  auto sB = make_tensor(make_smem_ptr(Bshm),
                        SmemLayoutB{}); // (kTileN, kTileK)

  // register, use tiled_mma to partition register A/B/C
  TiledMMA tiled_mma;
  auto thr_mma = tiled_mma.get_slice(idx);
  auto tCrA = thr_mma.partition_fragment_A(gA(_, _, 0)); // (MMA, MMA_M, MMA_K)
  auto tCrB = thr_mma.partition_fragment_B(gB(_, _, 0)); // (MMA, MMA_N, MMA_K)
  auto tCrD = thr_mma.partition_fragment_C(gD);          // (MMA, MMA_M, MMA_N)

  auto tCgD = thr_mma.partition_C(gD); // (MMA, MMA_M, MMA_N)
  // fill zero for accumulator
  clear(tCrD);

  // from global memory to shared memory
  G2SCopyA g2s_tiled_copy_a;
  auto g2s_thr_copy_a = g2s_tiled_copy_a.get_slice(idx);
  auto tAgA_copy = g2s_thr_copy_a.partition_S(gA); // (CPY, CPY_M, CPY_K, k)
  auto tAsA_copy =
      g2s_thr_copy_a.partition_D(sA); // (CPY, CPY_M, CPY_K)

  G2SCopyB g2s_tiled_copy_b;
  auto g2s_thr_copy_b = g2s_tiled_copy_b.get_slice(idx);
  auto tBgB_copy = g2s_thr_copy_b.partition_S(gB); // (CPY, CPY_N, CPY_K, k)
  auto tBsB_copy =
      g2s_thr_copy_b.partition_D(sB); // (CPY, CPY_N, CPY_K)

  // from shared memory to register, use tiled_mma to generate tiled_copy
  auto s2r_tiled_copy_a = make_tiled_copy_A(S2RCopyAtomA{}, tiled_mma);
  auto s2r_thr_copy_a = s2r_tiled_copy_a.get_slice(idx);
  auto tAsA = s2r_thr_copy_a.partition_S(sA);     // (CPY, CPY_M, CPY_K)
  auto tCrA_view = s2r_thr_copy_a.retile_D(tCrA); // (CPY, CPY_M, CPY_K)

  auto s2r_tiled_copy_b = make_tiled_copy_B(S2RCopyAtomB{}, tiled_mma);
  auto s2r_thr_copy_b = s2r_tiled_copy_b.get_slice(idx);
  auto tBsB = s2r_thr_copy_b.partition_S(sB);     // (CPY, CPY_N, CPY_K)
  auto tCrB_view = s2r_thr_copy_b.retile_D(tCrB); // (CPY, CPY_N, CPY_K)

  // loop over k: i. load tile, ii. mma
  int ntile = k / kTileK;
#pragma unroll 1
  for (int itile = 0; itile < ntile; ++itile)
  {
    // copy  (CPY, CPY_M, CPY_K) , async
    cute::copy(g2s_tiled_copy_a, tAgA_copy(_, _, _, itile),
               tAsA_copy(_, _, _));
    cute::copy(g2s_tiled_copy_b, tBgB_copy(_, _, _, itile),
               tBsB_copy(_, _, _));
    cp_async_fence();

    cp_async_wait<0>();
    __syncthreads();

    int nk = size<2>(tCrA);
#pragma unroll
    for (int ik = 0; ik < nk; ++ik)
    {
      // copy  (CPY, CPY_M), sync
      cute::copy(s2r_tiled_copy_a, tAsA(_, _, ik),
                 tCrA_view(_, _, ik));
      // copy  (CPY, CPY_N)
      cute::copy(s2r_tiled_copy_b, tBsB(_, _, ik),
                 tCrB_view(_, _, ik));
      // (MMA, MMA_M) x (MMA, MMA_N) => (MMA, MMA_M, MMA_N)
      cute::gemm(tiled_mma, tCrD, tCrA(_, _, ik), tCrB(_, _, ik), tCrD);
    } // for ik
  } // itile

  // register to global memory
  cute::copy(tCrD, tCgD);
}

// one-level pipeline
// global write opt
template <typename Config>
__global__ void
gemm_opt_p1(void *Dptr, const void *Aptr, const void *Bptr, int m, int n,
            int k)
{
  using T = typename Config::T;
  using SmemLayoutA = typename Config::SmemLayoutA;
  using SmemLayoutB = typename Config::SmemLayoutB;
  using TiledMMA = typename Config::MMA;

  using S2RCopyAtomA = typename Config::S2RCopyAtomA;
  using S2RCopyAtomB = typename Config::S2RCopyAtomB;
  using G2SCopyA = typename Config::G2SCopyA;
  using G2SCopyB = typename Config::G2SCopyB;

  constexpr int kTileM = Config::kTileM;
  constexpr int kTileN = Config::kTileN;
  constexpr int kTileK = Config::kTileK;
  constexpr int kStage = Config::kStage;

  extern __shared__ T shm_data[];

  T *Ashm = shm_data;
  T *Bshm = shm_data + cute::cosize(SmemLayoutA{});

  int idx = threadIdx.x;
  int ix = blockIdx.x;
  int iy = blockIdx.y;

  Tensor A = make_tensor(make_gmem_ptr((T *)Aptr), make_shape(m, k),
                         make_stride(k, Int<1>{})); // (M, K)
  Tensor B = make_tensor(make_gmem_ptr((T *)Bptr), make_shape(n, k),
                         make_stride(k, Int<1>{})); // (N, K)
  Tensor D = make_tensor(make_gmem_ptr((T *)Dptr), make_shape(m, n),
                         make_stride(n, Int<1>{})); // (M, N)
  // global memory
  Tensor gA = local_tile(A, make_tile(Int<kTileM>{}, Int<kTileK>{}),
                         make_coord(iy, _)); // (kTileM, kTileK, k)
  Tensor gB = local_tile(B, make_tile(Int<kTileN>{}, Int<kTileK>{}),
                         make_coord(ix, _)); // (kTileN, kTileK, k)
  Tensor gD = local_tile(D, make_tile(Int<kTileM>{}, Int<kTileN>{}),
                         make_coord(iy, ix)); // (kTileM, kTileN)

  // shared memory
  auto sA = make_tensor(make_smem_ptr(Ashm),
                        SmemLayoutA{}); // (kTileM, kTileK, kStage)
  auto sB = make_tensor(make_smem_ptr(Bshm),
                        SmemLayoutB{}); // (kTileN, kTileK, kStage)

  // register, use tiled_mma to partition register A/B/C
  TiledMMA tiled_mma;
  auto thr_mma = tiled_mma.get_slice(idx);
  auto tCrA = thr_mma.partition_fragment_A(gA(_, _, 0)); // (MMA, MMA_M, MMA_K)
  auto tCrB = thr_mma.partition_fragment_B(gB(_, _, 0)); // (MMA, MMA_N, MMA_K)
  auto tCrD = thr_mma.partition_fragment_C(gD);          // (MMA, MMA_M, MMA_N)
  // fill zero for accumulator
  clear(tCrD);

  auto tCgD = thr_mma.partition_C(gD); // (MMA, MMA_M, MMA_N)

  // from global memory to shared memory
  G2SCopyA g2s_tiled_copy_a;
  auto g2s_thr_copy_a = g2s_tiled_copy_a.get_slice(idx);
  auto tAgA_copy = g2s_thr_copy_a.partition_S(gA); // (CPY, CPY_M, CPY_K, k)
  auto tAsA_copy =
      g2s_thr_copy_a.partition_D(sA); // (CPY, CPY_M, CPY_K, kStage)

  G2SCopyB g2s_tiled_copy_b;
  auto g2s_thr_copy_b = g2s_tiled_copy_b.get_slice(idx);
  auto tBgB_copy = g2s_thr_copy_b.partition_S(gB); // (CPY, CPY_N, CPY_K, k)
  auto tBsB_copy =
      g2s_thr_copy_b.partition_D(sB); // (CPY, CPY_N, CPY_K, kStage)

  // from shared memory to register, use tiled_mma to generate tiled_copy
  auto s2r_tiled_copy_a = make_tiled_copy_A(S2RCopyAtomA{}, tiled_mma);
  auto s2r_thr_copy_a = s2r_tiled_copy_a.get_slice(idx);
  auto tAsA = s2r_thr_copy_a.partition_S(sA);     // (CPY, CPY_M, CPY_K, kStage)
  auto tCrA_view = s2r_thr_copy_a.retile_D(tCrA); // (CPY, CPY_M, CPY_K)

  auto s2r_tiled_copy_b = make_tiled_copy_B(S2RCopyAtomB{}, tiled_mma);
  auto s2r_thr_copy_b = s2r_tiled_copy_b.get_slice(idx);
  auto tBsB = s2r_thr_copy_b.partition_S(sB);     // (CPY, CPY_N, CPY_K, kStage)
  auto tCrB_view = s2r_thr_copy_b.retile_D(tCrB); // (CPY, CPY_N, CPY_K)

  // global -> shm, [0, k / kTileK]
  int itile_to_read = 0;
  // shm -> register, [0, kStage-1]
  int ismem_read = 0;
  // global -> shm, [0, kStage-1]
  int ismem_write = 0;

  // submit kStage - 1 tile
  // gmem -> shm
#pragma unroll
  for (int istage = 0; istage < kStage - 1; ++istage)
  {
    // copy  (CPY, CPY_M, CPY_K), asynchronous, thread-level
    copy(g2s_tiled_copy_a, tAgA_copy(_, _, _, istage),
         tAsA_copy(_, _, _, istage));
    copy(g2s_tiled_copy_b, tBgB_copy(_, _, _, istage),
         tBsB_copy(_, _, _, istage));
    cp_async_fence();

    ++itile_to_read;
    ++ismem_write;
  }

  // wait one submitted gmem->smem done
  cp_async_wait<kStage - 2>();
  // wait all threads in one warp complete
  __syncthreads();

  // loop over k: i. load tile, ii. mma
  int ntile = k / kTileK;

#pragma unroll 1
  for (int itile = 0; itile < ntile; ++itile)
  {
    int nk = size<2>(tCrA);

#pragma unroll
    for (int ik = 0; ik < nk; ++ik)
    {
      // shm -> reg s[itile][ik] -> r[ik]
      // copy  (CPY, CPY_M), use in next iteration ,sync
      cute::copy(s2r_tiled_copy_a, tAsA(_, _, ik, ismem_read),
                 tCrA_view(_, _, ik));
      cute::copy(s2r_tiled_copy_b, tBsB(_, _, ik, ismem_read),
                 tCrB_view(_, _, ik));
      // (MMA, MMA_M) x (MMA, MMA_N) => (MMA, MMA_M, MMA_N)
      cute::gemm(tiled_mma, tCrD, tCrA(_, _, ik), tCrB(_, _, ik), tCrD);

      if (ik == nk - 1)
      {
        cp_async_wait<kStage - 2>();
        __syncthreads();

        ismem_read = (ismem_read + 1) % kStage;
      }

      if (ik == 0)
      {
        if (itile_to_read < ntile)
        {
          // copy  (CPY, CPY_M, CPY_K)
          cute::copy(g2s_tiled_copy_a, tAgA_copy(_, _, _, itile_to_read),
                     tAsA_copy(_, _, _, ismem_write));
          cute::copy(g2s_tiled_copy_b, tBgB_copy(_, _, _, itile_to_read),
                     tBsB_copy(_, _, _, ismem_write));

          ++itile_to_read;
          ismem_write = (ismem_write + 1) % kStage;
        }

        cp_async_fence();
      }

    } // for ik
  } // itile

  // register to global memory
  cute::copy(tCrD, tCgD);
}

// two-level pipelne
template <typename Config>
__global__ void
gemm_opt_p2(void *Dptr, const void *Aptr, const void *Bptr, int m, int n,
            int k)
{
  using T = typename Config::T;
  using SmemLayoutA = typename Config::SmemLayoutA;
  using SmemLayoutB = typename Config::SmemLayoutB;
  using SmemLayoutC = typename Config::SmemLayoutC;
  using TiledMMA = typename Config::MMA;

  using S2RCopyAtomA = typename Config::S2RCopyAtomA;
  using S2RCopyAtomB = typename Config::S2RCopyAtomB;
  using G2SCopyA = typename Config::G2SCopyA;
  using G2SCopyB = typename Config::G2SCopyB;
  using R2SCopyAtomC = typename Config::R2SCopyAtomC;
  using S2GCopyAtomC = typename Config::S2GCopyAtomC;
  using S2GCopyC = typename Config::S2GCopyC;

  constexpr int kTileM = Config::kTileM;
  constexpr int kTileN = Config::kTileN;
  constexpr int kTileK = Config::kTileK;
  constexpr int kStage = Config::kStage;

  // max(A+B, C)
  extern __shared__ T shm_data[];

  T *Ashm = shm_data;
  T *Bshm = shm_data + cute::cosize(SmemLayoutA{});

  int idx = threadIdx.x;
  int ix = blockIdx.x;
  int iy = blockIdx.y;

  Tensor A = make_tensor(make_gmem_ptr((T *)Aptr), make_shape(m, k),
                         make_stride(k, Int<1>{})); // (M, K)
  Tensor B = make_tensor(make_gmem_ptr((T *)Bptr), make_shape(n, k),
                         make_stride(k, Int<1>{})); // (N, K)
  Tensor D = make_tensor(make_gmem_ptr((T *)Dptr), make_shape(m, n),
                         make_stride(n, Int<1>{})); // (M, N)
  // global memory
  Tensor gA = local_tile(A, make_tile(Int<kTileM>{}, Int<kTileK>{}),
                         make_coord(iy, _)); // (kTileM, kTileK, k)
  Tensor gB = local_tile(B, make_tile(Int<kTileN>{}, Int<kTileK>{}),
                         make_coord(ix, _)); // (kTileN, kTileK, k)
  Tensor gD = local_tile(D, make_tile(Int<kTileM>{}, Int<kTileN>{}),
                         make_coord(iy, ix)); // (kTileM, kTileN)

  // shared memory
  auto sA = make_tensor(make_smem_ptr(Ashm),
                        SmemLayoutA{}); // (kTileM, kTileK, kStage)
  auto sB = make_tensor(make_smem_ptr(Bshm),
                        SmemLayoutB{}); // (kTileN, kTileK, kStage)

  // register, use tiled_mma to partition register A/B/C
  TiledMMA tiled_mma;
  auto thr_mma = tiled_mma.get_slice(idx);
  auto tCrA = thr_mma.partition_fragment_A(gA(_, _, 0)); // (MMA, MMA_M, MMA_K)
  auto tCrB = thr_mma.partition_fragment_B(gB(_, _, 0)); // (MMA, MMA_N, MMA_K)
  auto tCrD = thr_mma.partition_fragment_C(gD);          // (MMA, MMA_M, MMA_N)
  // fill zero for accumulator
  clear(tCrD);

  auto tCgD = thr_mma.partition_C(gD); // (MMA, MMA_M, MMA_N)

  // from global memory to shared memory
  G2SCopyA g2s_tiled_copy_a;
  auto g2s_thr_copy_a = g2s_tiled_copy_a.get_slice(idx);
  auto tAgA_copy = g2s_thr_copy_a.partition_S(gA); // (CPY, CPY_M, CPY_K, k)
  auto tAsA_copy =
      g2s_thr_copy_a.partition_D(sA); // (CPY, CPY_M, CPY_K, kStage)

  G2SCopyB g2s_tiled_copy_b;
  auto g2s_thr_copy_b = g2s_tiled_copy_b.get_slice(idx);
  auto tBgB_copy = g2s_thr_copy_b.partition_S(gB); // (CPY, CPY_N, CPY_K, k)
  auto tBsB_copy =
      g2s_thr_copy_b.partition_D(sB); // (CPY, CPY_N, CPY_K, kStage)

  // from shared memory to register, use tiled_mma to generate tiled_copy
  auto s2r_tiled_copy_a = make_tiled_copy_A(S2RCopyAtomA{}, tiled_mma);
  auto s2r_thr_copy_a = s2r_tiled_copy_a.get_slice(idx);
  auto tAsA = s2r_thr_copy_a.partition_S(sA);     // (CPY, CPY_M, CPY_K, kStage)
  auto tCrA_view = s2r_thr_copy_a.retile_D(tCrA); // (CPY, CPY_M, CPY_K)

  auto s2r_tiled_copy_b = make_tiled_copy_B(S2RCopyAtomB{}, tiled_mma);
  auto s2r_thr_copy_b = s2r_tiled_copy_b.get_slice(idx);
  auto tBsB = s2r_thr_copy_b.partition_S(sB);     // (CPY, CPY_N, CPY_K, kStage)
  auto tCrB_view = s2r_thr_copy_b.retile_D(tCrB); // (CPY, CPY_N, CPY_K)

  // global -> shm, [0, k / kTileK]
  int itile_to_read = 0;
  // shm -> register, [0, kStage-1]
  int ismem_read = 0;
  // global -> shm, [0, kStage-1]
  int ismem_write = 0;

  // submit kStage - 1 tile
  // gmem -> shm
#pragma unroll
  for (int istage = 0; istage < kStage - 1; ++istage)
  {
    // copy  (CPY, CPY_M, CPY_K), asynchronous, thread-level
    copy(g2s_tiled_copy_a, tAgA_copy(_, _, _, istage),
         tAsA_copy(_, _, _, istage));
    copy(g2s_tiled_copy_b, tBgB_copy(_, _, _, istage),
         tBsB_copy(_, _, _, istage));
    cp_async_fence();

    ++itile_to_read;
    ++ismem_write;
  }

  // wait one submitted gmem->smem done
  cp_async_wait<kStage - 2>();
  // wait all threads in one warp complete
  __syncthreads();

  int ik = 0;
  // smem -> reg
  // copy  (CPY, CPY_M) ,sync
  copy(s2r_tiled_copy_a, tAsA(_, _, ik, ismem_read), tCrA_view(_, _, ik));
  // copy  (CPY, CPY_N) ,sync
  copy(s2r_tiled_copy_b, tBsB(_, _, ik, ismem_read), tCrB_view(_, _, ik));

  // loop over k: i. load tile, ii. mma
  int ntile = k / kTileK;

#pragma unroll 1
  for (int itile = 0; itile < ntile; ++itile)
  {
    int nk = size<2>(tCrA);

#pragma unroll
    for (int ik = 0; ik < nk; ++ik)
    {
      int ik_next = (ik + 1) % nk;

      if (ik == nk - 1)
      {
        cp_async_wait<kStage - 2>();
        __syncthreads();

        ismem_read = (ismem_read + 1) % kStage;
      }

      // shm -> reg s[itile][ik + 1] -> r[ik + 1]
      // copy  (CPY, CPY_M), use in next iteration ,sync
      cute::copy(s2r_tiled_copy_a, tAsA(_, _, ik_next, ismem_read),
                 tCrA_view(_, _, ik_next));
      cute::copy(s2r_tiled_copy_b, tBsB(_, _, ik_next, ismem_read),
                 tCrB_view(_, _, ik_next));

      if (ik == 0)
      {
        if (itile_to_read < ntile)
        {
          // copy  (CPY, CPY_M, CPY_K)
          cute::copy(g2s_tiled_copy_a, tAgA_copy(_, _, _, itile_to_read),
                     tAsA_copy(_, _, _, ismem_write));
          cute::copy(g2s_tiled_copy_b, tBgB_copy(_, _, _, itile_to_read),
                     tBsB_copy(_, _, _, ismem_write));

          ++itile_to_read;
          ismem_write = (ismem_write + 1) % kStage;
        }

        cp_async_fence();
      }
      // instruction reordering
      // (MMA, MMA_M) x (MMA, MMA_N) => (MMA, MMA_M, MMA_N)
      cute::gemm(tiled_mma, tCrD, tCrA(_, _, ik), tCrB(_, _, ik), tCrD);
    } // for ik
  } // itile

  // register to global memory
  cute::copy(tCrD, tCgD);
}

// global write opt
template <typename Config>
__global__ void
gemm_opt_final(void *Dptr, const void *Aptr, const void *Bptr, int m, int n,
               int k)
{
  using T = typename Config::T;
  using SmemLayoutA = typename Config::SmemLayoutA;
  using SmemLayoutB = typename Config::SmemLayoutB;
  using SmemLayoutC = typename Config::SmemLayoutC;
  using TiledMMA = typename Config::MMA;

  using S2RCopyAtomA = typename Config::S2RCopyAtomA;
  using S2RCopyAtomB = typename Config::S2RCopyAtomB;
  using G2SCopyA = typename Config::G2SCopyA;
  using G2SCopyB = typename Config::G2SCopyB;
  using R2SCopyAtomC = typename Config::R2SCopyAtomC;
  using S2GCopyAtomC = typename Config::S2GCopyAtomC;
  using S2GCopyC = typename Config::S2GCopyC;

  constexpr int kTileM = Config::kTileM;
  constexpr int kTileN = Config::kTileN;
  constexpr int kTileK = Config::kTileK;
  constexpr int kStage = Config::kStage;

  // max(A+B, C)
  extern __shared__ T shm_data[];

  T *Ashm = shm_data;
  T *Bshm = shm_data + cute::cosize(SmemLayoutA{});

  int idx = threadIdx.x;
  int ix = blockIdx.x;
  int iy = blockIdx.y;

  Tensor A = make_tensor(make_gmem_ptr((T *)Aptr), make_shape(m, k),
                         make_stride(k, Int<1>{})); // (M, K)
  Tensor B = make_tensor(make_gmem_ptr((T *)Bptr), make_shape(n, k),
                         make_stride(k, Int<1>{})); // (N, K)
  Tensor D = make_tensor(make_gmem_ptr((T *)Dptr), make_shape(m, n),
                         make_stride(n, Int<1>{})); // (M, N)
  // global memory
  Tensor gA = local_tile(A, make_tile(Int<kTileM>{}, Int<kTileK>{}),
                         make_coord(iy, _)); // (kTileM, kTileK, k)
  Tensor gB = local_tile(B, make_tile(Int<kTileN>{}, Int<kTileK>{}),
                         make_coord(ix, _)); // (kTileN, kTileK, k)
  Tensor gD = local_tile(D, make_tile(Int<kTileM>{}, Int<kTileN>{}),
                         make_coord(iy, ix)); // (kTileM, kTileN)

  // shared memory
  auto sA = make_tensor(make_smem_ptr(Ashm),
                        SmemLayoutA{}); // (kTileM, kTileK, kStage)
  auto sB = make_tensor(make_smem_ptr(Bshm),
                        SmemLayoutB{}); // (kTileN, kTileK, kStage)

  // register, use tiled_mma to partition register A/B/C
  TiledMMA tiled_mma;
  auto thr_mma = tiled_mma.get_slice(idx);
  auto tCrA = thr_mma.partition_fragment_A(gA(_, _, 0)); // (MMA, MMA_M, MMA_K)
  auto tCrB = thr_mma.partition_fragment_B(gB(_, _, 0)); // (MMA, MMA_N, MMA_K)
  auto tCrD = thr_mma.partition_fragment_C(gD);          // (MMA, MMA_M, MMA_N)
  // fill zero for accumulator
  clear(tCrD);

  // from global memory to shared memory
  G2SCopyA g2s_tiled_copy_a;
  auto g2s_thr_copy_a = g2s_tiled_copy_a.get_slice(idx);
  auto tAgA_copy = g2s_thr_copy_a.partition_S(gA); // (CPY, CPY_M, CPY_K, k)
  auto tAsA_copy =
      g2s_thr_copy_a.partition_D(sA); // (CPY, CPY_M, CPY_K, kStage)

  G2SCopyB g2s_tiled_copy_b;
  auto g2s_thr_copy_b = g2s_tiled_copy_b.get_slice(idx);
  auto tBgB_copy = g2s_thr_copy_b.partition_S(gB); // (CPY, CPY_N, CPY_K, k)
  auto tBsB_copy =
      g2s_thr_copy_b.partition_D(sB); // (CPY, CPY_N, CPY_K, kStage)

  // from shared memory to register, use tiled_mma to generate tiled_copy
  auto s2r_tiled_copy_a = make_tiled_copy_A(S2RCopyAtomA{}, tiled_mma);
  auto s2r_thr_copy_a = s2r_tiled_copy_a.get_slice(idx);
  auto tAsA = s2r_thr_copy_a.partition_S(sA);     // (CPY, CPY_M, CPY_K, kStage)
  auto tCrA_view = s2r_thr_copy_a.retile_D(tCrA); // (CPY, CPY_M, CPY_K)

  auto s2r_tiled_copy_b = make_tiled_copy_B(S2RCopyAtomB{}, tiled_mma);
  auto s2r_thr_copy_b = s2r_tiled_copy_b.get_slice(idx);
  auto tBsB = s2r_thr_copy_b.partition_S(sB);     // (CPY, CPY_N, CPY_K, kStage)
  auto tCrB_view = s2r_thr_copy_b.retile_D(tCrB); // (CPY, CPY_N, CPY_K)

  // global -> shm, [0, k / kTileK]
  int itile_to_read = 0;
  // shm -> register, [0, kStage-1]
  int ismem_read = 0;
  // global -> shm, [0, kStage-1]
  int ismem_write = 0;

  // submit kStage - 1 tile
  // gmem -> shm
#pragma unroll
  for (int istage = 0; istage < kStage - 1; ++istage)
  {
    // copy  (CPY, CPY_M, CPY_K), asynchronous, thread-level
    copy(g2s_tiled_copy_a, tAgA_copy(_, _, _, istage),
         tAsA_copy(_, _, _, istage));
    copy(g2s_tiled_copy_b, tBgB_copy(_, _, _, istage),
         tBsB_copy(_, _, _, istage));
    cp_async_fence();

    ++itile_to_read;
    ++ismem_write;
  }

  // wait one submitted gmem->smem done
  cp_async_wait<kStage - 2>();
  // wait all threads in one warp complete
  __syncthreads();

  int ik = 0;
  // smem -> reg
  // copy  (CPY, CPY_M) ,sync
  copy(s2r_tiled_copy_a, tAsA(_, _, ik, ismem_read), tCrA_view(_, _, ik));
  // copy  (CPY, CPY_N) ,sync
  copy(s2r_tiled_copy_b, tBsB(_, _, ik, ismem_read), tCrB_view(_, _, ik));

  // loop over k: i. load tile, ii. mma
  int ntile = k / kTileK;

#ifdef PRINT_INFO
  /*
    tCrA = thr_mma.partition_fragment_A(gA(_, _, 0))  (MMA, MMA_M, MMA_K) : ((_2,_2,_2),_4,_2)
    tAgA_copy = g2s_thr_copy_a.partition_S(gA)  (CPY, CPY_M, CPY_K, k) : ((_8,_1),_4,_1,32)
    tAsA_copy = g2s_thr_copy_a.partition_D(sA)  (CPY, CPY_M, CPY_K, kStage) : ((_8,_1),_4,_1,_3)
    tAsA = s2r_thr_copy_a.partition_S(sA)  (CPY, CPY_M, CPY_K, kStage) : ((_8,_1),_4,_2,_3)
    tCrA_view = s2r_thr_copy_a.retile_D(tCrA) (CPY, CPY_M, CPY_K) : ((_8,_1),_4,_2)
  */
  if (threadIdx.x == 0 && ix == 0 && iy == 0)
  {
    PRINT("tCrA = thr_mma.partition_fragment_A(gA(_, _, 0))  (MMA, MMA_M, MMA_K)", tCrA.shape());
    PRINT("tAgA_copy = g2s_thr_copy_a.partition_S(gA)  (CPY, CPY_M, CPY_K, k)", tAgA_copy.shape());
    PRINT("tAsA_copy = g2s_thr_copy_a.partition_D(sA)  (CPY, CPY_M, CPY_K, kStage)", tAsA_copy.shape());
    PRINT("tAsA = s2r_thr_copy_a.partition_S(sA)  (CPY, CPY_M, CPY_K, kStage)", tAsA.shape());
    PRINT("tCrA_view = s2r_thr_copy_a.retile_D(tCrA) (CPY, CPY_M, CPY_K)", tCrA_view.shape());
  }
#endif
#pragma unroll 1
  for (int itile = 0; itile < ntile; ++itile)
  {
    int nk = size<2>(tCrA);

#pragma unroll
    for (int ik = 0; ik < nk; ++ik)
    {
      int ik_next = (ik + 1) % nk;

      if (ik == nk - 1)
      {
        cp_async_wait<kStage - 2>();
        __syncthreads();

        ismem_read = (ismem_read + 1) % kStage;
      }

      // shm -> reg s[itile][ik + 1] -> r[ik + 1]
      // copy  (CPY, CPY_M), use in next iteration ,sync
      cute::copy(s2r_tiled_copy_a, tAsA(_, _, ik_next, ismem_read),
                 tCrA_view(_, _, ik_next));
      cute::copy(s2r_tiled_copy_b, tBsB(_, _, ik_next, ismem_read),
                 tCrB_view(_, _, ik_next));

      if (ik == 0)
      {
        if (itile_to_read < ntile)
        {
          // copy  (CPY, CPY_M, CPY_K)
          cute::copy(g2s_tiled_copy_a, tAgA_copy(_, _, _, itile_to_read),
                     tAsA_copy(_, _, _, ismem_write));
          cute::copy(g2s_tiled_copy_b, tBgB_copy(_, _, _, itile_to_read),
                     tBsB_copy(_, _, _, ismem_write));

          ++itile_to_read;
          ismem_write = (ismem_write + 1) % kStage;
        }

        cp_async_fence();
      }
      // instruction reordering
      // (MMA, MMA_M) x (MMA, MMA_N) => (MMA, MMA_M, MMA_N)
      cute::gemm(tiled_mma, tCrD, tCrA(_, _, ik), tCrB(_, _, ik), tCrD);
    } // for ik
  } // itile

  // use less shared memory as a scratchpad tile to use large wide instuction
  // Dreg -> shm -> reg -> global
  // (get<0>(MNK{}), get<1>(MNK{}), Int<kSmemLayoutCBatch>{})
  auto sC = make_tensor(sA(_, _, ismem_read).data(), SmemLayoutC{});

  auto r2s_tiled_copy_c = make_tiled_copy_C(R2SCopyAtomC{}, tiled_mma);
  auto r2s_thr_copy_c = r2s_tiled_copy_c.get_slice(idx);
  auto tCrC_r2s = r2s_thr_copy_c.retile_S(tCrD);  // (CPY, CPY_M, CPY_N)
  auto tCsC_r2s = r2s_thr_copy_c.partition_D(sC); // (CPY, _1, _1, pipe)

  S2GCopyC s2g_tiled_copy_c;
  auto s2g_thr_copy_c = s2g_tiled_copy_c.get_thread_slice(idx);
  auto tCsC_s2g = s2g_thr_copy_c.partition_S(sC); // (CPY, _1, _1, pipe)
  auto tCgC_s2g = s2g_thr_copy_c.partition_D(gD); // (CPY, CPY_M, CPY_N)

  auto tCgC_s2gx = group_modes<1, 3>(tCgC_s2g); // (CPY_, CPY_MN)
  auto tCrC_r2sx = group_modes<1, 3>(tCrC_r2s); // (CPY_, CPY_MN)

#ifdef PRINT_INFO
  /*
    sC : (_32,_32,_2)
    tCrC_r2s = r2s_thr_copy_c.retile_S(tCrD)  (CPY, CPY_M, CPY_N) : ((_2,(_2,_2)),_4,_4)
    tCsC_r2s = r2s_thr_copy_c.partition_D(sC)  (CPY, _1, _1, pipe) : ((_2,(_2,_2)),_1,_1,_2)
    tCsC_s2g = s2g_thr_copy_c.partition_S(sC)  (CPY, _1, _1, pipe) : ((_8,_1),_1,_1,_2)
    tCgC_s2g = s2g_thr_copy_c.partition_D(gD)  (CPY, CPY_M, CPY_N) : ((_8,_1),_4,_4)
    tCgC_s2gx = group_modes<1, 3>(tCgC_s2g) (CPY_, CPY_MN) : ((_8,_1),(_4,_4))
    tCrC_r2sx = group_modes<1, 3>(tCrC_r2s) (CPY_, CPY_MN) : ((_2,(_2,_2)),(_4,_4))
  */
  if (threadIdx.x == 0 && ix == 0 && iy == 0)
  {
    PRINT("sC", sC.shape());
    PRINT("tCrC_r2s = r2s_thr_copy_c.retile_S(tCrD)  (CPY, CPY_M, CPY_N)", tCrC_r2s.shape());
    PRINT("tCsC_r2s = r2s_thr_copy_c.partition_D(sC)  (CPY, _1, _1, pipe)", tCsC_r2s.shape());
    PRINT("tCsC_s2g = s2g_thr_copy_c.partition_S(sC)  (CPY, _1, _1, pipe)", tCsC_s2g.shape());
    PRINT("tCgC_s2g = s2g_thr_copy_c.partition_D(gD)  (CPY, CPY_M, CPY_N)", tCgC_s2g.shape());
    PRINT("tCgC_s2gx = group_modes<1, 3>(tCgC_s2g) (CPY_, CPY_MN)", tCgC_s2gx.shape());
    PRINT("tCrC_r2sx = group_modes<1, 3>(tCrC_r2s) (CPY_, CPY_MN)", tCrC_r2sx.shape());
  }
#endif

  int step = size<3>(tCsC_r2s); // pipe
#pragma unroll
  for (int i = 0; i < size<1>(tCrC_r2sx); i += step)
  {
    // reg -> shm
#pragma unroll
    for (int j = 0; j < step; ++j)
    {
      // we add a temp tensor to cope with accumulator and output data type
      // difference
      auto t = make_tensor_like<T>(tCrC_r2sx(_, i + j));
      // (_2,(_2,_2))
      cute::copy(tCrC_r2sx(_, i + j), t);
      // (_2,(_2,_2))
      cute::copy(r2s_tiled_copy_c, t, tCsC_r2s(_, 0, 0, j));
    }
    __syncthreads();

#pragma unroll
    // shm -> global
    for (int j = 0; j < step; ++j)
    {
      // (_8,_1)
      cute::copy(s2g_tiled_copy_c, tCsC_s2g(_, 0, 0, j), tCgC_s2gx(_, i + j));
    }

    __syncthreads();
  }
}

int main(int argc, char *argv[])
{
  std::string algo = "final";
  if (argc > 1)
  {
    algo = argv[1];
  }
  using T = cute::half_t;
  cudaEvent_t start, end;
  float elapsedTime;
  cudaEventCreate(&start);
  cudaEventCreate(&end);

  srand(1000);

  int M = 1024*64;
  int N = 128;
  int K = 1024;

  int count = 100;

  T *Aptr;
  T *Bptr;
  T *Dptr;
  cudaMalloc(&Aptr, sizeof(T) * M * K);
  cudaMalloc(&Bptr, sizeof(T) * N * K);
  cudaMalloc(&Dptr, sizeof(T) * M * N);

  T *Aptr_host;
  T *Bptr_host;
  Aptr_host = (T *)malloc(sizeof(T) * M * K);
  Bptr_host = (T *)malloc(sizeof(T) * N * K);
  auto tA = make_tensor(Aptr_host, make_shape(M, K), make_stride(K, 1));
  auto tB = make_tensor(Bptr_host, make_shape(N, K), make_stride(K, 1));
  cpu_rand_data(&tA);
  cpu_rand_data(&tB);
  cudaMemcpy(Aptr, Aptr_host, sizeof(T) * M * K, cudaMemcpyHostToDevice);
  cudaMemcpy(Bptr, Bptr_host, sizeof(T) * N * K, cudaMemcpyHostToDevice);

  // print(typename decltype(gemm_config)::MMA{});
  // print(typename decltype(gemm_config)::SmemLayoutA{});
  if (algo == "shm")
  {
    config::GemmConfigV1<T, 128, 128, 32> gemm_config;
    dim3 grid((N + gemm_config.kTileN - 1) / gemm_config.kTileN,
              (M + gemm_config.kTileM - 1) / gemm_config.kTileM);
    dim3 block(size(decltype(gemm_config)::MMA{}));
    int shm_size = gemm_config.kShmSize;

    cudaMemset(Dptr, 0, sizeof(T) * M * N);
    cudaFuncSetAttribute(gemm_opt_shm<decltype(gemm_config)>,
                         cudaFuncAttributeMaxDynamicSharedMemorySize, shm_size);

    cudaEventRecord(start);
    for (int it = 0; it < count; ++it)
    {
      gemm_opt_shm<decltype(gemm_config)>
          <<<grid, block, shm_size>>>(Dptr, Aptr, Bptr, M, N, K);
    }

    cudaEventRecord(end);
    cudaEventSynchronize(end);
    cudaEventElapsedTime(&elapsedTime, start, end);
    printf("gemm_opt_shm took %f ms.\n", elapsedTime / count);
  }
  else if (algo == "p1")
  {
    config::GemmConfig<T, 128, 128, 32, 3> gemm_config;
    dim3 grid((N + gemm_config.kTileN - 1) / gemm_config.kTileN,
              (M + gemm_config.kTileM - 1) / gemm_config.kTileM);
    dim3 block(size(decltype(gemm_config)::MMA{}));
    int shm_size = gemm_config.kShmSize;

    cudaMemset(Dptr, 0, sizeof(T) * M * N);
    cudaFuncSetAttribute(gemm_opt_p1<decltype(gemm_config)>,
                         cudaFuncAttributeMaxDynamicSharedMemorySize, shm_size);

    cudaEventRecord(start);
    for (int it = 0; it < count; ++it)
    {
      gemm_opt_p1<decltype(gemm_config)>
          <<<grid, block, shm_size>>>(Dptr, Aptr, Bptr, M, N, K);
    }

    cudaEventRecord(end);
    cudaEventSynchronize(end);
    cudaEventElapsedTime(&elapsedTime, start, end);
    printf("gemm_opt_p1 took %f ms.\n", elapsedTime / count);
  }
  else if (algo == "p2")
  {
    config::GemmConfig<T, 128, 128, 32, 3> gemm_config;
    dim3 grid((N + gemm_config.kTileN - 1) / gemm_config.kTileN,
              (M + gemm_config.kTileM - 1) / gemm_config.kTileM);
    dim3 block(size(decltype(gemm_config)::MMA{}));
    int shm_size = gemm_config.kShmSize;

    cudaMemset(Dptr, 0, sizeof(T) * M * N);
    cudaFuncSetAttribute(gemm_opt_p2<decltype(gemm_config)>,
                         cudaFuncAttributeMaxDynamicSharedMemorySize, shm_size);

    cudaEventRecord(start);
    for (int it = 0; it < count; ++it)
    {
      gemm_opt_p2<decltype(gemm_config)>
          <<<grid, block, shm_size>>>(Dptr, Aptr, Bptr, M, N, K);
    }

    cudaEventRecord(end);
    cudaEventSynchronize(end);
    cudaEventElapsedTime(&elapsedTime, start, end);
    printf("gemm_opt_p2 took %f ms.\n", elapsedTime / count);
  }
  else if (algo == "final")
  {
    config::GemmConfig<T, 128, 128, 32, 3> gemm_config;
    dim3 grid((N + gemm_config.kTileN - 1) / gemm_config.kTileN,
              (M + gemm_config.kTileM - 1) / gemm_config.kTileM);
    dim3 block(size(decltype(gemm_config)::MMA{}));
    int shm_size = gemm_config.kShmSize;

    cudaMemset(Dptr, 0, sizeof(T) * M * N);
    cudaFuncSetAttribute(gemm_opt_final<decltype(gemm_config)>,
                         cudaFuncAttributeMaxDynamicSharedMemorySize, shm_size);

    cudaEventRecord(start);
    for (int it = 0; it < count; ++it)
    {
      gemm_opt_final<decltype(gemm_config)>
          <<<grid, block, shm_size>>>(Dptr, Aptr, Bptr, M, N, K);
    }

    cudaEventRecord(end);
    cudaEventSynchronize(end);
    cudaEventElapsedTime(&elapsedTime, start, end);
    printf("gemm_opt_final took %f ms.\n", elapsedTime / count);
  }
}
