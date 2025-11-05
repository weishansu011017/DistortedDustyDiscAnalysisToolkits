"""
NeighborSearch

?????
"""
module NeighborSearch
using .Threads
using Statistics

include(joinpath(@__DIR__, "BinaryRadixTree", "MortonCode.jl"))


# Export function, marco, const...
for name in filter(s -> !startswith(string(s), "#"), names(@__MODULE__, all = true))
    if !startswith(String(name), "_") && (name != :eval) && (name != :include)
        @eval export $name
    end
end
end