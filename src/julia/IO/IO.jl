"""
IO

Handles data structures and input/output operations for PhantomRevealer.
This module provides:

- Definition of `ParticleDataFrame`, the main container for SPH data.
- Utility functions to add physical quantities to the data frame.
- Routines for reading and writing Phantom binary dumpfiles.

All I/O functions are built on top of `DataFrames.jl` for flexible data access.
"""
module IO

using .Threads 
using DataFrames
using HDF5
using PhantomRevealer


# IO & data structure
## ParticleDataFrame & basic adding quantities function
include(joinpath(@__DIR__,  "struct", "ParticleDataFrame.jl"))
include(joinpath(@__DIR__,  "struct", "add_quantities_prdf.jl"))

## GridDataset
include(joinpath(@__DIR__,  "struct", "GridDataset.jl"))

## Read & Write GridDataset
include(joinpath(@__DIR__, "gridsIO", "read_grids.jl"))
include(joinpath(@__DIR__, "gridsIO", "write_grids.jl"))

## Read & Write Phantom Binary dumpfiles
include(joinpath(@__DIR__, "phantomIO", "read_phantom.jl"))


# Export function, marco, const...
for name in filter(s -> !startswith(string(s), "#"), names(@__MODULE__, all = true))
    if !startswith(String(name), "_") && (name != :eval) && (name != :include)
        @eval export $name
    end
end
end