"""
IO

I/O utilities for Partia data structures.

This module provides a unified interface for reading and writing grid-based
datasets used in Partia.

The module aggregates lower-level I/O implementations and re-exports all
public-facing functions for external use.

# Included Components

## Grid Dataset I/O
- Reading grid-based datasets
- Writing grid-based datasets

Implemented in:
- `gridsIO/read_grids.jl`
- `gridsIO/write_grids.jl`
"""
module IO

using .Threads 
using HDF5
using Partia.Grids



# IO for data structure
## Read & Write GridDataset
include(joinpath(@__DIR__, "gridsIO", "read_grids.jl"))
include(joinpath(@__DIR__, "gridsIO", "write_grids.jl"))



# Export function, marco, const...
for name in filter(s -> !startswith(string(s), "#"), names(@__MODULE__, all = true))
    if !startswith(String(name), "_") && (name != :eval) && (name != :include)
        @eval export $name
    end
end
end
