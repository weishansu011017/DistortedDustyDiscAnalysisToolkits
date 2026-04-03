# ============================================================================ #
#  ParticleIO.jl - Test Suite Entry Point
# ============================================================================ #
#
#  Run with:  julia --project -e 'using Pkg; Pkg.test()'
#             or: include("test/runtests.jl") from the REPL
#
#  Ordering convention
#  -------------------
#  1. I/O layer    (Phantom binary reader)
#
# ============================================================================ #

using Test
using ParticleIO

# 1. I/O layer --------------------------------------------------------------- #
include("io_tests.jl")
