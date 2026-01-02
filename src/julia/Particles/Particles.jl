module Particles

using .Threads
using DataFrames
using Statistics

# ParticleDataFrame & basic adding quantities function
include(joinpath(@__DIR__, "ParticleDataFrame.jl"))
include(joinpath(@__DIR__, "add_quantities.jl"))


# Export function, marco, const...
for name in filter(s -> !startswith(string(s), "#"), names(@__MODULE__, all = true))
    if !startswith(String(name), "_") && (name != :eval) && (name != :include)
        @eval export $name
    end
end
end