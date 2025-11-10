"""
NeighborSearch

Provides spatial indexing structures and neighbor query routines for SPH data.
This module includes:
- Morton code generation and binary radix tree construction for particle ordering
- Linear bounding volume hierarchy (LBVH) builders and query kernels ready for CPU/GPU use
- Shared helper utilities exposed to interpolation and analysis pipelines

Implementations live under the `BinaryRadixTree/` and `LinearBVH/` directories.
"""
module NeighborSearch
using .Threads
using Statistics

# Binary radix tree
include(joinpath(@__DIR__, "BinaryRadixTree", "MortonCode.jl"))
include(joinpath(@__DIR__, "BinaryRadixTree", "BinaryRadixTree.jl"))

# Shared neighbor selection container
include(joinpath(@__DIR__, "NeighborSelection.jl"))

# Linear bounding vloume hierarchies (LBVH)
include(joinpath(@__DIR__, "LinearBVH", "LinearBVH.jl"))

# Export function, marco, const...
for name in filter(s -> !startswith(string(s), "#"), names(@__MODULE__, all = true))
    if !startswith(String(name), "_") && (name != :eval) && (name != :include)
        @eval export $name
    end
end
end