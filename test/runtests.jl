# ──────────────────────────────────────────────────────────────────────────── #
#  PhantomRevealer.jl — Test Suite Entry Point
# ──────────────────────────────────────────────────────────────────────────── #
#
#  Run with:  julia --project -e 'using Pkg; Pkg.test()'
#             or:  include("test/runtests.jl")  from the REPL
#
#  Ordering convention
#  ───────────────────
#  1. Utility functions             (Tools: EOS, coords, arrays)
#  2. I/O layer                     (Phantom binary reader)
#  3. Spatial data structures       (BRT → LBVH)
#  4. Kernel functions              (M4/M5/M6, Wendland C2/C4/C6)
#  5. Interpolation infrastructure  (constructors → traversal → physics)
#  6. Solvers / numerics            (TinyEigvals)
#  7. Physics modules               (Streaming Instability growth rate)
#
# ──────────────────────────────────────────────────────────────────────────── #

using Test
using PhantomRevealer

# ── 1. Utility functions ────────────────────────────────────────────── #
include("tools_tests.jl")

# ── 2. I/O layer ────────────────────────────────────────────────────── #
include("io_tests.jl")

# ── 3. Spatial data structures ──────────────────────────────────────── #
include("neighbor_search_tests.jl")

# ── 4. Kernel functions ─────────────────────────────────────────────── #
include("kernel_function_tests.jl")

# ── 5. Interpolation ────────────────────────────────────────────────── #
include("interpolation_test_common.jl")
include("interpolation_tests.jl")

# ── 6. Solvers / numerics ───────────────────────────────────────────── #
include("tiny_eigvals_tests.jl")

# ── 7. Physics modules ──────────────────────────────────────────────── #
include("growthrate_streaminginstability.jl")
