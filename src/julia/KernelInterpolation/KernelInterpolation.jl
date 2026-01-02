"""
KernelInterpolation

Provides core routines for kernel-based interpolation of SPH data.
This module includes:
- Kernel functions (spline, Wendland, and related forms)
- Line-of-sight (LOS) integration tables
- Grid-based interpolation and sampling utilities

All numerical implementations are organized under the `table/`,
`kernel_function/`, and `grid/` submodules.
"""
module KernelInterpolation
using .Threads 
using StaticArrays
using Adapt

using PhantomRevealer.Particles
using PhantomRevealer.Grids
using PhantomRevealer.NeighborSearch

# KernelInterpolation
include(joinpath(@__DIR__, "table", "los_tables.jl"))
## Kernels
include(joinpath(@__DIR__, "kernel_function", "kernel.jl"))
include(joinpath(@__DIR__, "kernel_function", "kernels", "M4_spline.jl"))
include(joinpath(@__DIR__, "kernel_function", "kernels", "M5_spline.jl"))
include(joinpath(@__DIR__, "kernel_function", "kernels", "M6_spline.jl"))
include(joinpath(@__DIR__, "kernel_function", "kernels", "C2_Wendland.jl"))
include(joinpath(@__DIR__, "kernel_function", "kernels", "C4_Wendland.jl"))
include(joinpath(@__DIR__, "kernel_function", "kernels", "C6_Wendland.jl"))
include(joinpath(@__DIR__, "kernel_function", "losintegrated_kernel.jl"))

## Single point interpolation
include(joinpath(@__DIR__, "interpolation_setup", "InterpolationStrategy.jl"))
include(joinpath(@__DIR__, "interpolation_setup", "InterpolationCatalog.jl"))
include(joinpath(@__DIR__, "interpolation_setup", "InterpolationInput.jl"))
include(joinpath(@__DIR__, "interpolation_setup", "constructor.jl"))

### LBVH Traversal
include(joinpath(@__DIR__, "single_point_interpolation", "accumulations", "scalar_accumulation.jl"))
include(joinpath(@__DIR__, "single_point_interpolation", "accumulations", "LOSscalar_accumulation.jl"))
include(joinpath(@__DIR__, "single_point_interpolation", "accumulations", "gradient_accumulation.jl"))
include(joinpath(@__DIR__, "single_point_interpolation", "accumulations", "divergence_accumulation.jl"))
include(joinpath(@__DIR__, "single_point_interpolation", "accumulations", "curl_accumulation.jl"))
include(joinpath(@__DIR__, "single_point_interpolation", "kernels", "scalar_interpolation.jl"))
include(joinpath(@__DIR__, "single_point_interpolation", "kernels", "LOSscalar_interpolation.jl"))
include(joinpath(@__DIR__, "single_point_interpolation", "kernels", "gradient_interpolation.jl"))
include(joinpath(@__DIR__, "single_point_interpolation", "kernels", "divergence_interpolation.jl"))
include(joinpath(@__DIR__, "single_point_interpolation", "kernels", "curl_interpolation.jl"))
include(joinpath(@__DIR__, "single_point_interpolation", "kernels", "general_interpolation.jl"))

## Grid interpolation
include(joinpath(@__DIR__, "grid_interpolation", "grid_interpolation.jl"))

# Export function, marco, const...
for name in filter(s -> !startswith(string(s), "#"), names(@__MODULE__, all = true))
    if !startswith(String(name), "_") && (name != :eval) && (name != :include)
        @eval export $name
    end
end
end