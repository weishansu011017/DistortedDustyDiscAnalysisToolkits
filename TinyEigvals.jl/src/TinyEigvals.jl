module TinyEigvals

using StaticArrays

# Add implementation includes here, for example:
include(joinpath(@__DIR__, "julia", "scaling", "scaling.jl"))
include(joinpath(@__DIR__, "julia", "balancing", "balancing.jl"))
include(joinpath(@__DIR__, "julia", "hessenberg_reduction", "hessenberg_reduction.jl"))
include(joinpath(@__DIR__, "julia", "schur_eigenvals", "schur_eigenvals.jl"))
include(joinpath(@__DIR__, "julia", "eigsolvers", "tiny_eigvals.jl"))

# Export function, marco, const...
for name in filter(s -> !startswith(string(s), "#"), names(@__MODULE__, all = true))
    if !startswith(String(name), "_") && (name != :eval) && (name != :include)
        @eval export $name
    end
end
end
