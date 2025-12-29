"""
IO

Handles data structures and input/output operations for PhantomRevealer.
This module provides:

- Definition of `ParticlesDataFrame`, the main container for SPH data.
- Utility functions to add physical quantities to the data frame.
- Routines for reading and writing Phantom binary dumpfiles.

All I/O functions are built on top of `DataFrames.jl` for flexible data access.
"""
module IO

using .Threads 
using DataFrames


# IO & data structure
## ParticlesDataFrame & basic adding quantities function
include(joinpath(@__DIR__,  "struct", "ParticlesDataFrame.jl"))
include(joinpath(@__DIR__,  "struct", "add_quantities_prdf.jl"))

## Read & Write Phantom Binary dumpfiles
include(joinpath(@__DIR__, "phantomIO", "read_phantom.jl"))


# Export function, marco, const...
for name in filter(s -> !startswith(string(s), "#"), names(@__MODULE__, all = true))
    if !startswith(String(name), "_") && (name != :eval) && (name != :include)
        @eval export $name
    end
end
end