"""
NeighborSearch

?????
"""
module NeighborSearch
using .Threads
using Statistics

# Binary radix tree
include(joinpath(@__DIR__, "BinaryRadixTree", "MortonCode.jl"))
include(joinpath(@__DIR__, "BinaryRadixTree", "BinaryRadixTree.jl"))

# Linear bounding vloume hierarchies (LBVH)
include(joinpath(@__DIR__, "LinearBVH", "LinearBVH.jl"))

# Export function, marco, const...
for name in filter(s -> !startswith(string(s), "#"), names(@__MODULE__, all = true))
    if !startswith(String(name), "_") && (name != :eval) && (name != :include)
        @eval export $name
    end
end
end