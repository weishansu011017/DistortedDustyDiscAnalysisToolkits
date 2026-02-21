module PhantomRevealer
# Include the Julia Module
using Pkg
using Reexport


# using Interpolations
##################### Core #####################
## Tools
include(joinpath(@__DIR__, "julia", "Tools", "Tools.jl"))
@reexport using .Tools

## Handwrite Eigenvalues solver
include(joinpath(@__DIR__, "julia", "BatchEigvals", "BatchEigvals.jl"))
@reexport using .BatchEigvals

## ParticleDataFrame
include(joinpath(@__DIR__, "julia", "Particles", "Particles.jl"))
@reexport using .Particles

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

## StreamingInstability
include(joinpath(@__DIR__, "julia", "StreamingInstability", "StreamingInstability.jl"))
@reexport using .StreamingInstability

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
    get_PhantomRevealer_path()
Get the folder of currently loaded PhantomRevealer

# Returns
- `String`: The folder of of currently loaded PhantomRevealer.
"""
function get_PhantomRevealer_path()
    return dirname(dirname(pathof(PhantomRevealer)))
end
end
