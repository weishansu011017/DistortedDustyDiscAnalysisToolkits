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
using PhantomRevealer
using Statistics
using CUDA


# Export function, marco, const...
for name in filter(s -> !startswith(string(s), "#"), names(@__MODULE__, all = true))
    if !startswith(String(name), "_") && (name != :eval) && (name != :include)
        @eval export $name
    end
end
end