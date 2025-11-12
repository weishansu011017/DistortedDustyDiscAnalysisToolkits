module AdaptStructure
using PhantomRevealer
using Metal
using Adapt

# InterpolationInput
include(joinpath(@__DIR__, "InterpolationInput.jl"))

# Export function, marco, const...
for name in filter(s -> !startswith(string(s), "#"), names(@__MODULE__, all = true))
    if !startswith(String(name), "_") && (name != :eval) && (name != :include)
        @eval export $name
    end
end
end