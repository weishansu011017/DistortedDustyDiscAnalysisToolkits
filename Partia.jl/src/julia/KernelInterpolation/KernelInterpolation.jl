"""
KernelInterpolation

Core infrastructure for kernel-based interpolation of SPH data in Partia.

This module implements the full interpolation pipeline used to map particle-based
SPH quantities onto arbitrary points or grids. It is designed to be:

- Numerically explicit and allocation-aware
- Compatible with GPU execution via `Adapt.jl`
- Structured around LBVH-based neighbor traversal

# Scope and Responsibilities

The module provides:

## Kernel Definitions
- Spline kernels (M4, M5, M6)
- Wendland kernels (C2, C4, C6)
- line integrated kernels

Kernel implementations are located under:
- `kernel_function/`
- `kernel_function/kernels/`

## Line Integration Tables
- Precomputed tables for line-integrated kernels
- Used by projected / column-integrated quantities

Implemented under:
- `table/`

## Interpolation Framework
- Strategy and catalog abstractions for interpolation modes
- Strongly-typed, GPU-safe interpolation input definitions

Implemented under:
- `interpolation_setup/`

## Single-Point Interpolation
- Scalar interpolation
- Line-integrated scalar interpolation
- Gradient, divergence, and curl evaluation
- General interpolation kernels

All single-point interpolation routines are implemented using
LBVH-based neighbor traversal and accumulation, under:
- `single_point_interpolation/`

## Grid-Based Interpolation
- Sampling SPH data onto structured grids

Implemented under:
- `grid_interpolation/`
"""
module KernelInterpolation
using .Threads 
using StaticArrays
using Adapt

using Partia.Grids
using Partia.NeighborSearch

# KernelInterpolation
include(joinpath(@__DIR__, "table", "line_integrated_kernel_tables.jl"))
## Kernels
include(joinpath(@__DIR__, "kernel_function", "kernel.jl"))
include(joinpath(@__DIR__, "kernel_function", "kernels", "M4_spline.jl"))
include(joinpath(@__DIR__, "kernel_function", "kernels", "M5_spline.jl"))
include(joinpath(@__DIR__, "kernel_function", "kernels", "M6_spline.jl"))
include(joinpath(@__DIR__, "kernel_function", "kernels", "C2_Wendland.jl"))
include(joinpath(@__DIR__, "kernel_function", "kernels", "C4_Wendland.jl"))
include(joinpath(@__DIR__, "kernel_function", "kernels", "C6_Wendland.jl"))
include(joinpath(@__DIR__, "kernel_function", "line_integrated_kernel.jl"))

## Execution backends
include(joinpath(@__DIR__, "ExecutionBackend", "AbstractExecutionBackend.jl"))

## Single point interpolation
include(joinpath(@__DIR__, "interpolation_setup", "InterpolationStrategy.jl"))
include(joinpath(@__DIR__, "interpolation_setup", "InterpolationCatalog.jl"))
include(joinpath(@__DIR__, "interpolation_setup", "InterpolationInput.jl"))
include(joinpath(@__DIR__, "interpolation_setup", "constructor.jl"))

### LBVH Traversal
#### Point interpolations
include(joinpath(@__DIR__, "single_point_interpolation", "accumulations", "scalar_accumulation.jl"))
include(joinpath(@__DIR__, "single_point_interpolation", "accumulations", "gradient_accumulation.jl"))
include(joinpath(@__DIR__, "single_point_interpolation", "accumulations", "divergence_accumulation.jl"))
include(joinpath(@__DIR__, "single_point_interpolation", "accumulations", "curl_accumulation.jl"))
include(joinpath(@__DIR__, "single_point_interpolation", "kernels", "scalar_interpolation.jl"))
include(joinpath(@__DIR__, "single_point_interpolation", "kernels", "gradient_interpolation.jl"))
include(joinpath(@__DIR__, "single_point_interpolation", "kernels", "divergence_interpolation.jl"))
include(joinpath(@__DIR__, "single_point_interpolation", "kernels", "curl_interpolation.jl"))
include(joinpath(@__DIR__, "single_point_interpolation", "kernels", "general_interpolation.jl"))

#### Line integrated interpolations
include(joinpath(@__DIR__, "line_integrated_interpolation", "accumulations", "line_integrated_scalar_accumulation.jl"))
include(joinpath(@__DIR__, "line_integrated_interpolation", "kernels", "line_integrated_scalar_interpolation.jl"))

## Grid interpolation
include(joinpath(@__DIR__, "grid_interpolation", "initialize_interpolation.jl"))
include(joinpath(@__DIR__, "grid_interpolation", "PointSamples_interpolation.jl"))
include(joinpath(@__DIR__, "grid_interpolation", "StructuredGrid_interpolation.jl"))
include(joinpath(@__DIR__, "grid_interpolation", "LineSamples_interpolation.jl"))

# Export function, marco, const...
for name in filter(s -> !startswith(string(s), "#"), names(@__MODULE__, all = true))
    if !startswith(String(name), "_") && (name != :eval) && (name != :include)
        @eval export $name
    end
end
end
