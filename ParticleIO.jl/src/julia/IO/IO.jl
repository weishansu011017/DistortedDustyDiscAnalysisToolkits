"""
IO

I/O utilities for particle data structures.

This module provides a unified interface for reading particle-based simulation
data, including:

- Phantom binary dump files (particle-based data)

The module aggregates lower-level I/O implementations and re-exports all
public-facing functions for external use.

# Included Components

## Phantom Binary I/O
- Reading Phantom binary dump files

Implemented in:
- `phantomIO/read_phantom.jl`

"""
module IO

using .Threads
using DataFrames
using ParticleIO.Particles


# IO for data structure
## Read Phantom Binary dumpfiles
include(joinpath(@__DIR__, "phantomIO", "read_phantom.jl"))

# Export function, marco, const...
for name in filter(s -> !startswith(string(s), "#"), names(@__MODULE__, all = true))
    if !startswith(String(name), "_") && (name != :eval) && (name != :include)
        @eval export $name
    end
end
end
