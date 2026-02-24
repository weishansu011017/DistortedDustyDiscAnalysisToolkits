"""
TinyEigvals

Implements a pure-Julia, fixed-size eigenvalue solver for tiny complex matrices
(typically N ≤ 15), designed for high-throughput and allocation-free workflows.
This module is a simplified, small-N adaptation of the standard LAPACK
eigensolver pipeline (e.g. scaling, balancing, Hessenberg reduction, and Schur
eigenvalue extraction), but implemented entirely in Julia without calling LAPACK.

The implementation targets `StaticArrays` (`MMatrix`/`SMatrix`) and is intended
as a lightweight alternative to `LinearAlgebra.eigvals` when the matrix size is
small and many independent eigenproblems must be solved in batch.
"""
module TinyEigvals
using StaticArrays

using PhantomRevealer.Tools

# Pure-static eigensolver for tiny complex matrices (N ≤ 15)
include(joinpath(@__DIR__, "scaling", "scaling.jl"))
include(joinpath(@__DIR__, "balancing", "balancing.jl"))
include(joinpath(@__DIR__, "hessenberg_reduction", "hessenberg_reduction.jl"))
include(joinpath(@__DIR__, "schur_eigenvals", "schur_eigenvals.jl"))
include(joinpath(@__DIR__, "eigsolvers", "tiny_eigvals.jl"))

# Export function, marco, const...
for name in filter(s -> !startswith(string(s), "#"), names(@__MODULE__, all = true))
    if !startswith(String(name), "_") && (name != :eval) && (name != :include)
        @eval export $name
    end
end
end