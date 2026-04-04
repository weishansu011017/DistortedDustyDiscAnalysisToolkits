module Partia
# Include the Julia Module
using Logging
using Pkg
using Reexport

##################### Core #####################
## Tools
include(joinpath(@__DIR__, "julia", "Tools", "Tools.jl"))
@reexport using .Tools

## Structure of Grids
include(joinpath(@__DIR__, "julia", "Grids", "Grids.jl"))
@reexport using .Grids

## IO & data structure
include(joinpath(@__DIR__, "julia", "IO", "IO.jl"))
@reexport using .IO

## NeighborSearch 
include(joinpath(@__DIR__, "julia", "NeighborSearch", "NeighborSearch.jl"))
@reexport using .NeighborSearch

## KernelInterpolation
include(joinpath(@__DIR__, "julia", "KernelInterpolation", "KernelInterpolation.jl"))
@reexport using .KernelInterpolation

##################### CUDA Extension #####################
# Dummy file (similar to a C/C++ header) used to declare the module interface.
# Enables precompilation even when CUDA is not available.
# Named with .jlh to prevent accidental inclusion at runtime.
include(joinpath(@__DIR__, "julia", "ExtDummy", "CUDAExtDummy.jlh"))        
@reexport using .CUDAExtDummy

#################### Metal Extension #####################
# Dummy file (similar to a C/C++ header) used to declare the module interface.
# Enables precompilation even when CUDA is not available.
# Named with .jlh to prevent accidental inclusion at runtime.
include(joinpath(@__DIR__, "julia", "ExtDummy", "MetalExtDummy.jlh"))        
@reexport using .MetalExtDummy

################################################


# Initialize function
"""
    get_Partia_path()
Get the folder of currently loaded Partia

# Returns
- `String`: The folder of of currently loaded Partia.
"""
function get_Partia_path()
    return dirname(dirname(pathof(Partia)))
end

# Package metadata helpers.
version() = pkgversion(@__MODULE__)

function about()
    @info "Partia analysis Module\n  Version: $(version())\n  Made by Wei-Shan Su, Apr 2026"
    return nothing
end
end
