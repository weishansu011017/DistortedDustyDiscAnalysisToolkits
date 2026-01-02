"""
Grids

Grid data abstractions and representations for PhantomRevealer.

This module defines the grid-side data structures used for sampling,
storing, and manipulating gridded quantities derived from SPH data.
It provides a unified interface for both structured and unstructured
grid representations, together with dataset-level containers.

# Scope and Responsibilities

The module provides:

## Coordinate System Definitions
- Flags and utilities for identifying grid coordinate systems
  (e.g. Cartesian, cylindrical, spherical)

Implemented in:
- `grids/coordinate.jl`

## Core Grid Abstractions
- `AbstractGrid`, the common interface for all grid types
- `GeneralGrid`, a flexible grid representation with explicit axes
- `StructuredGrid`, a regular grid with implicit topology

Implemented in:
- `grids/AbstractGrid.jl`
- `grids/GeneralGrid.jl`
- `grids/StructuredGrid.jl`

## Grid Transformations
- Conversion utilities between `StructuredGrid` and `GeneralGrid`
- Used to bridge regular grids and more general representations

Implemented in:
- `grids/transform.jl`

## Grid Dataset Containers
- `GridBundle`, a lightweight container for grouped grid objects
- `GridDataset`, a dataset-level abstraction for gridded fields

Implemented in:
- `griddataset/GridBundle.jl`
- `griddataset/GridDataset.jl`
"""
module Grids
using .Threads
using Statistics
using Adapt

# Flag of coordinate system
include(joinpath(@__DIR__, "grids", "coordinate.jl"))

# AbstractGrid
include(joinpath(@__DIR__, "grids", "AbstractGrid.jl"))

# GeneralGrid
include(joinpath(@__DIR__, "grids", "GeneralGrid.jl"))

# StructuredGrid
include(joinpath(@__DIR__, "grids", "StructuredGrid.jl"))

# Transfromation between StructuredGrid and GeneralGrid
include(joinpath(@__DIR__, "grids", "transform.jl"))

# GridBundle
include(joinpath(@__DIR__,  "griddataset", "GridBundle.jl"))

# GridDataset
include(joinpath(@__DIR__,  "griddataset", "GridDataset.jl"))

# Export function, marco, const...
for name in filter(s -> !startswith(string(s), "#"), names(@__MODULE__, all = true))
    if !startswith(String(name), "_") && (name != :eval) && (name != :include)
        @eval export $name
    end
end
end