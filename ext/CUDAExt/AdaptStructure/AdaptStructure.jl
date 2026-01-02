module AdaptStructure
using PhantomRevealer
using CUDA
using Adapt

# to_CuVector()
include(joinpath(@__DIR__, "to_CuVector.jl"))

# to_HostVector()
include(joinpath(@__DIR__, "to_HostVector.jl"))

# Export function, marco, const...
for name in filter(s -> !startswith(string(s), "#"), names(@__MODULE__, all = true))
    if !startswith(String(name), "_") && (name != :eval) && (name != :include)
        @eval export $name
    end
end
end