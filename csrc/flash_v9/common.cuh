#pragma once

// Shared types, layouts, and helpers for Flash v9 kernels.
// Populated as forward / backward / double_backward are implemented in
// commits 3-5. Built on CUTLASS 3 / CuTe primitives.

namespace flash_v9 {

// Reserved for:
//   - Layout traits parameterized on (Br, Bc, headdim)
//   - CuTe tile descriptors and copy/mma atoms
//   - Online softmax state (m, l running stats)
//   - SM80 cp.async copy traits

} // namespace flash_v9
