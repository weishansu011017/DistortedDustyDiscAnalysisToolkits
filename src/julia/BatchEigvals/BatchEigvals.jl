module BatchEigvals
using .Threads
using StaticArrays
using LinearAlgebra

# Pure-static eigensolver for tiny complex matrices (N ≤ 15)
include(joinpath(@__DIR__, "scaling", "scaling.jl"))
include(joinpath(@__DIR__, "balancing", "balancing.jl"))
include(joinpath(@__DIR__, "hessenberg_reduction", "hessenberg_reduction.jl"))
include(joinpath(@__DIR__, "schur_eigenvals", "schur_eigenvals.jl"))
include(joinpath(@__DIR__, "eigsolvers", "beigvals.jl"))

# Export function, marco, const...
for name in filter(s -> !startswith(string(s), "#"), names(@__MODULE__, all = true))
    if !startswith(String(name), "_") && (name != :eval) && (name != :include)
        @eval export $name
    end
end
end