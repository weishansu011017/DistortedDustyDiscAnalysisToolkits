"""
Particles

Core particle data abstractions for Partia.

This module defines the canonical particle-level data structures and
associated utilities used throughout Partia. It provides a
DataFrame-backed representation of SPH particle data together with
helper routines for constructing and augmenting particle quantities.

# Scope and Responsibilities

The module provides:

## Particle Data Container
- `ParticleDataFrame`, a structured wrapper around `DataFrames.DataFrame`
  tailored for SPH particle data
- Standardised column conventions for particle properties
  (e.g. position, velocity, mass, density)

Implemented in:
- `ParticleDataFrame.jl`

## Quantity Construction Utilities
- Functions for adding derived or auxiliary particle quantities
- Designed to operate in-place on `ParticleDataFrame` objects
- Supports statistical operations where appropriate

Implemented in:
- `add_quantities.jl`
"""
module Particles

using Base.Threads
using DataFrames
using Statistics
using Partia.Tools

# ParticleDataFrame & basic adding quantities function
include(joinpath(@__DIR__, "ParticleDataFrame.jl"))
include(joinpath(@__DIR__, "add_quantities.jl"))


# Export function, marco, const...
for name in filter(s -> !startswith(string(s), "#"), names(@__MODULE__, all = true))
    if !startswith(String(name), "_") && (name != :eval) && (name != :include)
        @eval export $name
    end
end
end
