#pragma once

// Shared types, layouts, and helpers for Flash v9 kernels.
// Built on CUTLASS 4.x / CuTe primitives.
//
// Reference: Dao-AILab/flash-attention v2.8.3 (hopper/mainloop_fwd_sm80.hpp).
// We mirror the same swizzled-smem + cp.async + SM80 MMA patterns but strip
// out the GQA / varlen / paged / rotary machinery FA needs and we do not.

#include <cute/tensor.hpp>
#include <cute/atom/copy_atom.hpp>
#include <cute/atom/mma_atom.hpp>
#include <cutlass/cutlass.h>
#include <cutlass/numeric_types.h>
#include <cutlass/numeric_conversion.h>
#include <cutlass/array.h>
#include <cuda_bf16.h>
#include <cuda_fp16.h>

namespace flash_v9 {

// CuTe smoke test (kept from commit 2): forces template instantiation so the
// build fails if CUTLASS headers are not picked up.
__host__ __device__ inline constexpr int cute_smoke_cosize() {
    return cute::size(cute::make_layout(
        cute::make_shape(cute::Int<8>{}, cute::Int<16>{})
    ));
}
static_assert(cute_smoke_cosize() == 128,
              "CuTe smoke test failed: CUTLASS headers not picked up");

// ----------------------------------------------------------------------------
// Gmem copy traits.
// We load Q / K / V from gmem with 16-byte (int4) vectorized cp.async copies.
// kBlockKGmem is the contiguous chunk size in the K (headdim) direction,
// chosen so each "row" of the smem-atom is 128 bytes (one cache line).
// ----------------------------------------------------------------------------
template <int Headdim, typename Element>
struct GmemCopyTraits {
    static constexpr int kBytesPerElement = sizeof(Element);
    static constexpr int kBytesPerRow     = Headdim * kBytesPerElement;
    static constexpr int kBlockKGmem      =
        (kBytesPerRow % 128 == 0) ? (128 / kBytesPerElement) :
        (kBytesPerRow %  64 == 0) ? ( 64 / kBytesPerElement) :
                                    ( 32 / kBytesPerElement);
    static constexpr int kSwizzle =
        (kBlockKGmem == 128) ? 4 :
        (kBlockKGmem ==  64) ? 3 :
        (kBlockKGmem ==  32) ? 2 : 1;
    static constexpr int kSwizzleBase =
        (kBytesPerElement == 4) ? 2 :
        (kBytesPerElement == 2) ? 3 : 4;
    static constexpr int kElemsPerLoad = 16 / kBytesPerElement;
};

// ----------------------------------------------------------------------------
// Smem layout atom: swizzled 8 x kBlockKGmem tile.
// The swizzle composes with the row-major inner layout so that adjacent
// threads accessing the same row do not hit the same smem bank.
// ----------------------------------------------------------------------------
template <int Headdim, typename Element>
using SmemLayoutAtomQKV = decltype(cute::composition(
    cute::Swizzle<
        GmemCopyTraits<Headdim, Element>::kSwizzle,
        GmemCopyTraits<Headdim, Element>::kSwizzleBase,
        GmemCopyTraits<Headdim, Element>::kSwizzleBase
    >{},
    cute::Layout<
        cute::Shape <cute::_8, cute::Int<GmemCopyTraits<Headdim, Element>::kBlockKGmem>>,
        cute::Stride<     cute::Int<GmemCopyTraits<Headdim, Element>::kBlockKGmem>, cute::_1>
    >{}
));

// SmemLayoutQ: (Br, Headdim) tiling of the atom.
template <int Br, int Headdim, typename Element>
using SmemLayoutQ = decltype(cute::tile_to_shape(
    SmemLayoutAtomQKV<Headdim, Element>{},
    cute::make_shape(cute::Int<Br>{}, cute::Int<Headdim>{})
));

// SmemLayoutK / SmemLayoutV: (Bc, Headdim) tiling of the atom.
template <int Bc, int Headdim, typename Element>
using SmemLayoutK = decltype(cute::tile_to_shape(
    SmemLayoutAtomQKV<Headdim, Element>{},
    cute::make_shape(cute::Int<Bc>{}, cute::Int<Headdim>{})
));

template <int Bc, int Headdim, typename Element>
using SmemLayoutV = SmemLayoutK<Bc, Headdim, Element>;

// SmemLayoutVt: transposed view of sV for the P.V MMA's B operand. The
// physical bytes stay where the cp.async wrote them (row-major (Bc, Headdim))
// but we view them as (Headdim, Bc) so partition_fragment_B sees the right
// (N=Headdim, K=Bc) shape and SM75_U16x8_LDSM_T can transpose-load.
template <int Bc, int Headdim, typename Element>
using SmemLayoutVt = decltype(cute::composition(
    SmemLayoutV<Bc, Headdim, Element>{},
    cute::make_ordered_layout(
        cute::make_shape(cute::Int<Headdim>{}, cute::Int<Bc>{}),
        cute::Step<cute::_2, cute::_1>{}
    )
));

// SmemLayoutKt: same construction as SmemLayoutVt; used wherever K is the
// B-operand and the MMA expects its K-axis to be the second smem dim.
template <int Bc, int Headdim, typename Element>
using SmemLayoutKt = SmemLayoutVt<Bc, Headdim, Element>;

// SmemLayoutQt: transposed view of sQ for the dK MMA's B operand
// (dK = dS^T . Q, where Q is consumed as (Headdim, Br) for B).
template <int Br, int Headdim, typename Element>
using SmemLayoutQt = decltype(cute::composition(
    SmemLayoutQ<Br, Headdim, Element>{},
    cute::make_ordered_layout(
        cute::make_shape(cute::Int<Headdim>{}, cute::Int<Br>{}),
        cute::Step<cute::_2, cute::_1>{}
    )
));

// SmemLayoutdOt: same as SmemLayoutQt (dO is the same shape as Q).
template <int Br, int Headdim, typename Element>
using SmemLayoutdOt = SmemLayoutQt<Br, Headdim, Element>;

// ----------------------------------------------------------------------------
// SmemLayoutPdS: smem layout for staging P and dS between MMAs in the
// backward and double-backward kernels. Shape is (Br, Bc), swizzled the
// same way as SmemLayoutAtomQKV but parameterised on Bc (which is the
// inner stride for the staged tile) instead of Headdim.
// ----------------------------------------------------------------------------
template <int Bc, typename Element>
using SmemLayoutAtomPdS = decltype(cute::composition(
    cute::Swizzle<
        GmemCopyTraits<Bc, Element>::kSwizzle,
        GmemCopyTraits<Bc, Element>::kSwizzleBase,
        GmemCopyTraits<Bc, Element>::kSwizzleBase
    >{},
    cute::Layout<
        cute::Shape <cute::_8, cute::Int<GmemCopyTraits<Bc, Element>::kBlockKGmem>>,
        cute::Stride<     cute::Int<GmemCopyTraits<Bc, Element>::kBlockKGmem>, cute::_1>
    >{}
));

template <int Br, int Bc, typename Element>
using SmemLayoutPdS = decltype(cute::tile_to_shape(
    SmemLayoutAtomPdS<Bc, Element>{},
    cute::make_shape(cute::Int<Br>{}, cute::Int<Bc>{})
));

// SmemLayoutPdSt: transposed view of sPdS for using P^T or dS^T as the
// A-operand of a subsequent MMA via SM75_U16x8_LDSM_T.
template <int Br, int Bc, typename Element>
using SmemLayoutPdSt = decltype(cute::composition(
    SmemLayoutPdS<Br, Bc, Element>{},
    cute::make_ordered_layout(
        cute::make_shape(cute::Int<Bc>{}, cute::Int<Br>{}),
        cute::Step<cute::_2, cute::_1>{}
    )
));

// ----------------------------------------------------------------------------
// Gmem TiledCopy descriptor: SM80 cp.async with int4 (16B = 8 bf16) vectors.
// Layout-of-threads is chosen so each row of the smem atom is loaded by a
// contiguous group of threads.
// ----------------------------------------------------------------------------
template <int Headdim, int NumThreads, typename Element>
struct GmemTiledCopyTraits {
    using Traits = GmemCopyTraits<Headdim, Element>;
    static constexpr int kGmemThreadsPerRow = Traits::kBlockKGmem / Traits::kElemsPerLoad;
    static_assert(NumThreads % kGmemThreadsPerRow == 0,
                  "NumThreads must be a multiple of kGmemThreadsPerRow");
    using GmemLayoutAtom = cute::Layout<
        cute::Shape <cute::Int<NumThreads / kGmemThreadsPerRow>, cute::Int<kGmemThreadsPerRow>>,
        cute::Stride<cute::Int<kGmemThreadsPerRow>,              cute::_1>
    >;
    using GmemCopyAtom = cute::Copy_Atom<
        cute::SM80_CP_ASYNC_CACHEGLOBAL<cute::uint128_t>,
        Element
    >;
    using GmemTiledCopy = decltype(cute::make_tiled_copy(
        GmemCopyAtom{},
        GmemLayoutAtom{},
        cute::Layout<cute::Shape<cute::_1, cute::Int<Traits::kElemsPerLoad>>>{}
    ));
};

// ----------------------------------------------------------------------------
// SM80 MMA atom: 16x8x16 with fp32 accumulation. bf16 inputs use the BF16
// variant; fp16 inputs use the F16 variant.
// TiledMma_SM80: NumWarps warps along M, single warp along N and K.
// One warp does an mma_16x16x16 instance per tile; NumWarps warps cover
// M = 16 * NumWarps rows.
// ----------------------------------------------------------------------------
template <typename Element>
using MmaAtom_SM80 = cute::MMA_Atom<
    std::conditional_t<
        std::is_same_v<Element, cutlass::half_t>,
        cute::SM80_16x8x16_F32F16F16F32_TN,
        cute::SM80_16x8x16_F32BF16BF16F32_TN
    >
>;

template <typename Element, int NumWarps>
using TiledMma_SM80 = cute::TiledMMA<
    MmaAtom_SM80<Element>,
    cute::Layout<cute::Shape<cute::Int<NumWarps>, cute::_1, cute::_1>>,
    cute::Tile<cute::Int<16 * NumWarps>, cute::_16, cute::_16>
>;

// ----------------------------------------------------------------------------
// Gmem TiledCopy for the output store (smem -> gmem). cp.async is gmem->smem
// only, so we use a regular vectorized store atom on the way out.
// ----------------------------------------------------------------------------
template <int Headdim, int NumThreads, typename Element>
struct GmemTiledCopyOTraits {
    using Traits = GmemCopyTraits<Headdim, Element>;
    static constexpr int kGmemThreadsPerRow = Traits::kBlockKGmem / Traits::kElemsPerLoad;
    using GmemLayoutAtom = cute::Layout<
        cute::Shape <cute::Int<NumThreads / kGmemThreadsPerRow>, cute::Int<kGmemThreadsPerRow>>,
        cute::Stride<cute::Int<kGmemThreadsPerRow>,              cute::_1>
    >;
    using GmemCopyAtom = cute::Copy_Atom<
        cute::AutoVectorizingCopyWithAssumedAlignment<128>,
        Element
    >;
    using GmemTiledCopy = decltype(cute::make_tiled_copy(
        GmemCopyAtom{},
        GmemLayoutAtom{},
        cute::Layout<cute::Shape<cute::_1, cute::Int<Traits::kElemsPerLoad>>>{}
    ));
};

// ----------------------------------------------------------------------------
// Smem usage in bytes for Q + K + V. sO reuses the sQ region for the
// output epilogue (Q is no longer needed by the time we store O).
// ----------------------------------------------------------------------------
template <int Br, int Bc, int Headdim, typename Element>
struct SmemSize {
    static constexpr int sQ_elems = cute::cosize_v<SmemLayoutQ<Br, Headdim, Element>>;
    static constexpr int sK_elems = cute::cosize_v<SmemLayoutK<Bc, Headdim, Element>>;
    static constexpr int sV_elems = cute::cosize_v<SmemLayoutV<Bc, Headdim, Element>>;
    static constexpr int total_bytes = (sQ_elems + sK_elems + sV_elems) * sizeof(Element);
};

// ----------------------------------------------------------------------------
// Online softmax helpers (cribbed from flash-attention/hopper/softmax.h +
// utils.h, simplified for our fixed-seqlen non-fp8 path).
// ----------------------------------------------------------------------------

template <typename T>
struct MaxOp {
    __device__ __forceinline__ T operator()(T const& x, T const& y) { return x > y ? x : y; }
};
template <>
struct MaxOp<float> {
    __device__ __forceinline__ float operator()(float const& x, float const& y) { return max(x, y); }
};

template <typename T>
struct SumOp {
    __device__ __forceinline__ T operator()(T const& x, T const& y) { return x + y; }
};

template <int THREADS>
struct Allreduce {
    static_assert(THREADS == 32 || THREADS == 16 || THREADS == 8 || THREADS == 4);
    template <typename T, typename Operator>
    static __device__ __forceinline__ T run(T x, Operator& op) {
        constexpr int OFFSET = THREADS / 2;
        x = op(x, __shfl_xor_sync(uint32_t(-1), x, OFFSET));
        return Allreduce<OFFSET>::run(x, op);
    }
};
template <>
struct Allreduce<2> {
    template <typename T, typename Operator>
    static __device__ __forceinline__ T run(T x, Operator& op) {
        x = op(x, __shfl_xor_sync(uint32_t(-1), x, 1));
        return x;
    }
};

// SM80: convert acc_layout from (MMA=4, MMA_M, MMA_N) to (nrow=(2, MMA_M), ncol=(2, MMA_N)).
template <typename Layout>
CUTE_DEVICE auto convert_layout_acc_rowcol(Layout acc_layout) {
    using namespace cute;
    static_assert(decltype(size<0>(acc_layout))::value == 4);
    static_assert(decltype(rank(acc_layout))::value == 3);
    auto l = logical_divide(acc_layout, Shape<_2>{});
    return make_layout(make_layout(get<0, 1>(l), get<1>(l)),
                       make_layout(get<0, 0>(l), get<2>(l)));
}

// SM80 m16n8k16: convert acc_layout from (MMA=4, MMA_M, MMA_N) to ((4, 2), MMA_M, MMA_N/2)
// so the fp32 accumulator can be re-interpreted as an A-operand fragment for
// the next MMA (P.V).
template <typename TiledMma, typename Layout>
CUTE_DEVICE auto convert_layout_acc_Aregs(Layout acc_layout) {
    using namespace cute;
    using X = Underscore;
    static_assert(decltype(size<0>(acc_layout))::value == 4);
    static_assert(decltype(rank(acc_layout))::value == 3);
    constexpr int mma_shape_K = get<2>(typename TiledMma::Shape_MNK{});
    static_assert(mma_shape_K == 8 || mma_shape_K == 16);
    if constexpr (mma_shape_K == 8) {
        return acc_layout;
    } else {
        auto l = logical_divide(acc_layout, Shape<X, X, _2>{});
        return make_layout(make_layout(get<0>(l), get<2, 0>(l)),
                           get<1>(l),
                           get<2, 1>(l));
    }
}

// fp32 -> bf16/fp16 in-register conversion using cutlass::NumericArrayConverter
// for proper vectorized PTX (cvt.bf16x2.f32x2 etc.).
template <typename Engine, typename Layout, typename EngineOut>
CUTE_DEVICE void convert_type_out(cute::Tensor<Engine, Layout> const& tensor,
                                  cute::Tensor<EngineOut, Layout>& out) {
    using namespace cute;
    using From = typename Engine::value_type;
    using To   = typename EngineOut::value_type;
    static constexpr int FragmentSize =
        std::max(sizeof(From) / sizeof(To), sizeof(To) / sizeof(From));
    static_assert(CUTE_STATIC_V(size(tensor)) % FragmentSize == 0,
                  "Fragment size does not vectorize properly");
    Tensor frag    = recast<cutlass::Array<From, FragmentSize> const>(tensor);
    Tensor out_frg = recast<cutlass::Array<To,   FragmentSize>>(out);
    cutlass::NumericArrayConverter<To, From, FragmentSize> op;
    CUTE_UNROLL
    for (int i = 0; i < size(frag); ++i) { out_frg[i] = op(frag[i]); }
}

// Per-thread reduction over a 2D (rows, cols) view: reduce along the col
// dimension to produce a per-row summary.
template <bool zero_init = true, typename T0, typename L0, typename T1, typename L1, typename Op>
__device__ __forceinline__ void thread_reduce_(cute::Tensor<T0, L0> const& tensor,
                                               cute::Tensor<T1, L1>& summary,
                                               Op& op) {
    using namespace cute;
    static_assert(L0::rank == 2);
    static_assert(L1::rank == 1);
    CUTE_STATIC_ASSERT_V(size<0>(summary) == size<0>(tensor));
    CUTE_UNROLL
    for (int ni = 0; ni < size<1>(tensor); ++ni) {
        CUTE_UNROLL
        for (int mi = 0; mi < size<0>(tensor); ++mi) {
            summary(mi) = (zero_init && ni == 0) ? tensor(mi, ni) : op(summary(mi), tensor(mi, ni));
        }
    }
}

// Warp-quad allreduce: each row of an SM80 16x8 MMA tile is shared by 4
// consecutive lanes (lanes 0..3 share one row pair, 4..7 share the next, ...).
template <typename T0, typename L0, typename T1, typename L1, typename Op>
__device__ __forceinline__ void quad_allreduce_(cute::Tensor<T0, L0>& dst,
                                                cute::Tensor<T1, L1>& src,
                                                Op& op) {
    using namespace cute;
    CUTE_STATIC_ASSERT_V(size(dst) == size(src));
    CUTE_UNROLL
    for (int i = 0; i < size(dst); ++i) { dst(i) = Allreduce<4>::run(src(i), op); }
}

template <bool zero_init = true, typename T0, typename L0, typename T1, typename L1>
__device__ __forceinline__ void reduce_max(cute::Tensor<T0, L0> const& tensor,
                                           cute::Tensor<T1, L1>& mx) {
    MaxOp<float> op;
    thread_reduce_<zero_init>(tensor, mx, op);
    quad_allreduce_(mx, mx, op);
}

template <bool zero_init = true, bool warp_reduce = false,
          typename T0, typename L0, typename T1, typename L1>
__device__ __forceinline__ void reduce_sum(cute::Tensor<T0, L0> const& tensor,
                                           cute::Tensor<T1, L1>& sm) {
    SumOp<float> op;
    thread_reduce_<zero_init>(tensor, sm, op);
    if constexpr (warp_reduce) { quad_allreduce_(sm, sm, op); }
}

template <bool Check_inf = true,
          typename T0, typename L0, typename T1, typename L1>
__device__ __forceinline__ void scale_apply_exp2(cute::Tensor<T0, L0>& tensor,
                                                 cute::Tensor<T1, L1> const& mx,
                                                 float scale_log2) {
    using namespace cute;
    static_assert(L0::rank == 2);
    static_assert(L1::rank == 1);
    CUTE_STATIC_ASSERT_V(size<0>(mx) == size<0>(tensor));
    CUTE_UNROLL
    for (int mi = 0; mi < size<0>(tensor); ++mi) {
        const float max_scaled = Check_inf
            ? (mx(mi) == -INFINITY ? 0.f : mx(mi) * scale_log2)
            : mx(mi) * scale_log2;
        CUTE_UNROLL
        for (int ni = 0; ni < size<1>(tensor); ++ni) {
            tensor(mi, ni) = exp2f(tensor(mi, ni) * scale_log2 - max_scaled);
        }
    }
}

// Online softmax state: per-thread running row_max and row_sum across
// successive K/V tiles. kNRows must match size<0>(convert_layout_acc_rowcol(rS)),
// i.e. 2 * MMA_M for SM80.
template <int kNRows>
struct Softmax {
    using TensorT = decltype(cute::make_tensor<float>(cute::Shape<cute::Int<kNRows>>{}));
    TensorT row_max, row_sum;
    float const softmax_scale_log2;

    __device__ Softmax(float scale_log2) : softmax_scale_log2(scale_log2) {}

    template <bool Is_first, bool Check_inf = false, typename TensorAcc>
    __forceinline__ __device__ TensorT max_get_scale(TensorAcc& acc_s) {
        using namespace cute;
        Tensor scores = make_tensor(acc_s.data(), convert_layout_acc_rowcol(acc_s.layout()));
        static_assert(decltype(size<0>(scores))::value == kNRows);
        TensorT scores_scale;
        if constexpr (Is_first) {
            reduce_max</*zero_init=*/true>(scores, row_max);
            cute::fill(scores_scale, 1.f);
        } else {
            Tensor prev_max = cute::make_fragment_like(row_max);
            cute::copy(row_max, prev_max);
            reduce_max</*zero_init=*/false>(scores, row_max);
            CUTE_UNROLL
            for (int mi = 0; mi < size(row_max); ++mi) {
                float cur = !Check_inf ? row_max(mi) : (row_max(mi) == -INFINITY ? 0.f : row_max(mi));
                scores_scale(mi) = exp2f((prev_max(mi) - cur) * softmax_scale_log2);
                row_sum(mi) *= scores_scale(mi);
            }
        }
        return scores_scale;
    }

    template <bool Is_first, bool Check_inf = false, typename TensorAcc>
    __forceinline__ __device__ void online_softmax(TensorAcc& acc_s) {
        using namespace cute;
        Tensor scores = make_tensor(acc_s.data(), convert_layout_acc_rowcol(acc_s.layout()));
        scale_apply_exp2<Check_inf>(scores, row_max, softmax_scale_log2);
        reduce_sum</*zero_init=*/Is_first, /*warp_reduce=*/false>(scores, row_sum);
    }

    __forceinline__ __device__ TensorT finalize(float final_scale = 1.f) {
        SumOp<float> op;
        quad_allreduce_(row_sum, row_sum, op);
        TensorT scores_scale;
        CUTE_UNROLL
        for (int mi = 0; mi < size(row_sum); ++mi) {
            float s = row_sum(mi);
            float inv = (s == 0.f || s != s) ? 0.f : 1.f / s;
            scores_scale(mi) = inv * final_scale;
            // Store logsumexp = max * ln(2) * log2(e) + log(sum) = max + log(sum)
            // because softmax_scale_log2 already absorbs the softmax scale.
            row_sum(mi) = (s == 0.f || s != s)
                ? -INFINITY
                : row_max(mi) * (softmax_scale_log2 * float(M_LN2)) + __logf(s);
        }
        return scores_scale;
    }

    template <typename TensorO>
    __forceinline__ __device__ void rescale_o(TensorO& acc_o, TensorT const& scores_scale) {
        using namespace cute;
        Tensor acc_o_rc = make_tensor(acc_o.data(), convert_layout_acc_rowcol(acc_o.layout()));
        static_assert(decltype(size<0>(acc_o_rc))::value == kNRows);
        CUTE_UNROLL
        for (int mi = 0; mi < size<0>(acc_o_rc); ++mi) {
            CUTE_UNROLL
            for (int ni = 0; ni < size<1>(acc_o_rc); ++ni) { acc_o_rc(mi, ni) *= scores_scale(mi); }
        }
    }
};

// ----------------------------------------------------------------------------
// Reusable per-tile helpers for backward and double backward.
// ----------------------------------------------------------------------------

// P = exp(rS * scale - L_thread), in place via exp2.
//   exp(S*scale - L) = exp2((S*scale - L) * log2(e))
//                    = exp2(S * scale_log2 - L * log2(e))
// where scale_log2 = scale * log2(e). The saved L is in "scaled S" units
// (L = logsumexp(S*scale) from the forward), so we multiply L by log2(e)
// (not by scale_log2) when shifting into exp2 input space.
template <typename T0, typename L0, typename T1, typename L1>
__device__ __forceinline__ void apply_lse_exp2(
    cute::Tensor<T0, L0>& rS_rc,
    cute::Tensor<T1, L1> const& L_thread,
    float scale_log2
) {
    using namespace cute;
    static_assert(L0::rank == 2);
    static_assert(L1::rank == 1);
    CUTE_STATIC_ASSERT_V(size<0>(rS_rc) == size<0>(L_thread));
    constexpr float kLog2E = 1.4426950408889634f;
    CUTE_UNROLL
    for (int mi = 0; mi < size<0>(rS_rc); ++mi) {
        const float L_log2 = L_thread(mi) * kLog2E;
        CUTE_UNROLL
        for (int ni = 0; ni < size<1>(rS_rc); ++ni) {
            rS_rc(mi, ni) = exp2f(rS_rc(mi, ni) * scale_log2 - L_log2);
        }
    }
}

// dS = P * (dP - D_thread) * scale, in place on rdP_rc (which then holds dS).
template <typename TP, typename LP, typename TdP, typename LdP, typename TD, typename LD>
__device__ __forceinline__ void apply_dS(
    cute::Tensor<TP, LP> const& rP_rc,
    cute::Tensor<TdP, LdP>& rdP_rc,
    cute::Tensor<TD, LD> const& D_thread,
    float scale
) {
    using namespace cute;
    static_assert(LP::rank == 2);
    static_assert(LdP::rank == 2);
    static_assert(LD::rank == 1);
    CUTE_UNROLL
    for (int mi = 0; mi < size<0>(rdP_rc); ++mi) {
        const float Di = D_thread(mi);
        CUTE_UNROLL
        for (int ni = 0; ni < size<1>(rdP_rc); ++ni) {
            rdP_rc(mi, ni) = rP_rc(mi, ni) * (rdP_rc(mi, ni) - Di) * scale;
        }
    }
}

// Apply causal mask to a (Br, Bc) S tile: k_idx > q_idx -> -INFINITY.
// q_off = q_block * Br, k_off = kv_block * Bc.
// Templated on the TiledMma so the per-thread (row, col) coords can be derived
// from partition_C of an identity tensor (this matches the same layout the
// forward kernel uses).
template <int Br, int Bc, typename TiledMma, typename T, typename L>
__device__ __forceinline__ void causal_mask_tile(
    cute::Tensor<T, L>& rS, int q_off, int k_off, int thread_idx
) {
    using namespace cute;
    auto thr_mma = TiledMma{}.get_thread_slice(thread_idx);
    auto cS = make_identity_tensor(Shape<Int<Br>, Int<Bc>>{});
    auto tScS = thr_mma.partition_C(cS);
    auto tScS_rc = make_tensor(tScS.data(),
                               convert_layout_acc_rowcol(tScS.layout()));
    auto rS_rc   = make_tensor(rS.data(),
                               convert_layout_acc_rowcol(rS.layout()));
    CUTE_UNROLL
    for (int m = 0; m < size<0>(rS_rc); ++m) {
        const int q_idx = q_off + get<0>(tScS_rc(m, _0{}));
        CUTE_UNROLL
        for (int n = 0; n < size<1>(rS_rc); ++n) {
            const int k_idx = k_off + get<1>(tScS_rc(_0{}, n));
            if (k_idx > q_idx) rS_rc(m, n) = -INFINITY;
        }
    }
}

// Build per-thread L (logsumexp) and D (rowsum(dO*O)) register tensors
// from gmem [B,H,N] strided memory. Each thread owns 2 rows of an SM80
// 16x8 MMA tile (the standard SM80 16x8x16 C-partition).
template <int Br, int kNRows = 2>
__device__ __forceinline__ void load_L_D_per_thread(
    const float* L_bh, const float* D_bh,
    int q_block, int N, int warp_id, int lane_id,
    float* L_out, float* D_out
) {
    const int row_lo = warp_id * 16 + (lane_id >> 2);
    const int row_hi = row_lo + 8;
    const int q_row_lo = q_block * Br + row_lo;
    const int q_row_hi = q_block * Br + row_hi;
    L_out[0] = (q_row_lo < N) ? L_bh[q_row_lo] : 0.f;
    L_out[1] = (q_row_hi < N) ? L_bh[q_row_hi] : 0.f;
    if (D_bh != nullptr) {
        D_out[0] = (q_row_lo < N) ? D_bh[q_row_lo] : 0.f;
        D_out[1] = (q_row_hi < N) ? D_bh[q_row_hi] : 0.f;
    }
}

} // namespace flash_v9
