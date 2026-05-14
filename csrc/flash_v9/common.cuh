#pragma once

// Shared types, layouts, and helpers for Flash v9 kernels.
// Built on CUTLASS 4.x / CuTe primitives.

#include <cute/tensor.hpp>

namespace flash_v9 {

// CuTe smoke test: compile-time use of CuTe layout algebra. Forces the
// compiler to instantiate CuTe templates so the build fails if CUTLASS
// headers are not picked up by setup.py's include_dirs.
//
// The static_assert below verifies that the resulting layout has the
// expected cosize (8 * 16 = 128). Triggered at compile time by every TU
// that includes this header.
__host__ __device__ inline constexpr int cute_smoke_cosize() {
    return cute::size(cute::make_layout(
        cute::make_shape(cute::Int<8>{}, cute::Int<16>{})
    ));
}

static_assert(cute_smoke_cosize() == 128,
              "CuTe smoke test failed: CUTLASS headers not picked up");

// Reserved for:
//   - Layout traits parameterized on (Br, Bc, headdim)
//   - CuTe tile descriptors and copy/mma atoms
//   - Online softmax state (m, l running stats)
//   - SM80 cp.async copy traits

} // namespace flash_v9
