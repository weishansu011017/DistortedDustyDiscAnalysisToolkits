"""
ParticleIO

Particle-oriented front-end utilities built on top of Partia.

This package collects the particle-side data model and I/O workflow used for
SPH analysis, while delegating core numerical interpolation and grid-side
infrastructure to `Partia`.

The module currently provides:

- `Particles`: `ParticleDataFrame` and particle convenience operations
- `IO`: Phantom dump-file reading utilities
- `PartiaAdapter`: particle-to-interpolation adapters built on top of `Partia`

`ParticleIO` is intended to serve as the particle-data entry layer for workflows
that eventually feed into `Partia`'s interpolation and analysis machinery.
"""
module ParticleIO

# Include the Julia Module
using Reexport

## Particle data structures and convenience operations
include(joinpath(@__DIR__, "julia", "Particles", "Particles.jl"))
@reexport using .Particles

## Particle-side I/O
include(joinpath(@__DIR__, "julia", "IO", "IO.jl"))
@reexport using .IO

## Adapters from particle containers to Partia interpolation inputs
include(joinpath(@__DIR__, "julia", "PartiaAdapter", "PartiaAdapter.jl"))
@reexport using .PartiaAdapter

# Export function, macro, const...
for name in filter(s -> !startswith(string(s), "#"), names(@__MODULE__, all = true))
    if !startswith(String(name), "_") && (name != :eval) && (name != :include)
        @eval export $name
    end
end
end
