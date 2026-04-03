# ──────────────────────────────────────────────────────────────────────────── #
#  Partia.jl - Test Suite Entry Point
# ──────────────────────────────────────────────────────────────────────────── #
#
#  Run with:  julia --project -e 'using Pkg; Pkg.test()'
#             or: include("test/runtests.jl") from the REPL
#
#  Ordering convention
#  ───────────────────
#  1. Utility functions             (Tools: EOS, coords, arrays)
#  2. I/O layer                     (grid I/O)
#  3. Spatial data structures       (BRT, LBVH)
#  4. Kernel functions              (M4/M5/M6, Wendland C2/C4/C6)
#  5. Interpolation infrastructure  (constructors, traversal, physics)
#  6. Additional core checks        (type stability, traversal)
#
# ──────────────────────────────────────────────────────────────────────────── #

using Test
using Partia

# 1. Utility functions ------------------------------------------------------ #
include("tools_tests.jl")

# 2. Spatial data structures ------------------------------------------------ #
include("neighbor_search_tests.jl")

# 3. Kernel functions ------------------------------------------------------- #
include("kernel_function_tests.jl")

# 4. Interpolation ---------------------------------------------------------- #
include("interpolation_test_common.jl")
include("interpolation_tests.jl")
include("grid_interpolation_tests.jl")
include("traversal_analytic.jl")
include("type_stability.jl")
