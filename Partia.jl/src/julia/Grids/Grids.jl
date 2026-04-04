"""
Grids

Grid data abstractions and representations for Partia.

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
- `AbstractSamples`, the common interface for sample-based grid types
- `PointSamples`, a flexible sample representation with explicit coordinates
- `LineSamples`, a flexible line-sample representation with explicit origins and directions
- `StructuredGrid`, a regular grid with implicit topology

Implemented in:
- `grids/AbstractGrid.jl`
- `grids/AbstractSamples.jl`
- `grids/PointSamples.jl`
- `grids/LineSamples.jl`
- `grids/StructuredGrid.jl`

## Grid Transformations
- Conversion utilities between `StructuredGrid` and `PointSamples`
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
using ..Tools: _cylin2cart, _sph2cart

# Flag of coordinate system
include(joinpath(@__DIR__, "grids", "coordinate.jl"))

# AbstractGrid
include(joinpath(@__DIR__, "grids", "AbstractGrid.jl"))

# AbstractSamples
include(joinpath(@__DIR__, "grids", "AbstractSamples.jl"))

# PointSamples
include(joinpath(@__DIR__, "grids", "PointSamples.jl"))

# LineSamples
include(joinpath(@__DIR__, "grids", "LineSamples.jl"))

# StructuredGrid
include(joinpath(@__DIR__, "grids", "StructuredGrid.jl"))

# Transfromation between StructuredGrid and PointSamples
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
