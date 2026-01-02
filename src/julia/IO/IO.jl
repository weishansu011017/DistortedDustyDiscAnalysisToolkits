"""
IO

I/O utilities for PhantomRevealer data structures.

This module provides a unified interface for reading and writing simulation data
used in PhantomRevealer, including:

- Phantom binary dump files (particle-based data)
- Grid-based datasets stored in HDF5 format

The module aggregates lower-level I/O implementations and re-exports all
public-facing functions for external use.

# Included Components

## Phantom Binary I/O
- Reading Phantom binary dump files
- Writing Phantom binary dump files

Implemented in:
- `phantomIO/read_phantom.jl`

## Grid Dataset I/O
- Reading grid-based datasets
- Writing grid-based datasets

Implemented in:
- `gridsIO/read_grids.jl`
- `gridsIO/write_grids.jl`
"""
module IO

using .Threads 
using DataFrames
using HDF5
using PhantomRevealer.Particles
using PhantomRevealer.Grids



# IO for data structure
## Read & Write Phantom Binary dumpfiles
include(joinpath(@__DIR__, "phantomIO", "read_phantom.jl"))

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