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
        std::is_same_v<Element, __half>,
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
// Smem usage in bytes for one Q tile + one K tile + one V tile.
// Used by the kernel launcher to allocate dynamic shared memory.
// ----------------------------------------------------------------------------
template <int Br, int Bc, int Headdim, typename Element>
struct SmemSize {
    static constexpr int sQ_elems = cute::cosize_v<SmemLayoutQ<Br, Headdim, Element>>;
    static constexpr int sK_elems = cute::cosize_v<SmemLayoutK<Bc, Headdim, Element>>;
    static constexpr int sV_elems = cute::cosize_v<SmemLayoutV<Bc, Headdim, Element>>;
    static constexpr int total_bytes = (sQ_elems + sK_elems + sV_elems) * sizeof(Element);
};

} // namespace flash_v9
