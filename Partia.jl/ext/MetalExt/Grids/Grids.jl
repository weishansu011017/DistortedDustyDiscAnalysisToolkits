module Grids
using Partia
using Metal

# Grid construction
include(joinpath(@__DIR__, "grids", "PointSamples.jl"))
include(joinpath(@__DIR__, "grids", "LineSamples.jl"))


# Export function, marco, const...
for name in filter(s -> !startswith(string(s), "#"), names(@__MODULE__, all = true))
    if !startswith(String(name), "_") && (name != :eval) && (name != :include)
        @eval export $name
    end
end
end
