module PhantomRevealer
# Include the Julia Module
using Pkg
using Reexport


# using Interpolations
##################### Core #####################
## Tools
include(joinpath(@__DIR__, "julia", "Tools", "Tools.jl"))
@reexport using .Tools

# include(joinpath(@__DIR__, "julia", "Tools", "eos_properties.jl"))
# include(joinpath(@__DIR__, "julia", "Tools", "logging.jl"))
# include(joinpath(@__DIR__, "julia", "Tools", "coordinate_transformations.jl"))
# include(joinpath(@__DIR__, "julia", "Tools", "array_operations.jl"))

## IO & data structure
include(joinpath(@__DIR__, "julia", "IO", "IO.jl"))
@reexport using .IO

## NeighborSearch 
include(joinpath(@__DIR__, "julia", "NeighborSearch", "NeighborSearch.jl"))
@reexport using .NeighborSearch

## KernelInterpolation
include(joinpath(@__DIR__, "julia", "KernelInterpolation", "KernelInterpolation.jl"))
@reexport using .KernelInterpolation
# include(joinpath(@__DIR__, "julia", "KernelInterpolation", "table", "los_tables.jl"))
# include(joinpath(@__DIR__, "julia", "KernelInterpolation", "kernel_function.jl"))
# include(joinpath(@__DIR__, "julia", "KernelInterpolation", "grid.jl"))


### Constructors of the `InterpolationInput` from `PhantomRevealerDataFrame`
include(joinpath(@__DIR__, "julia", "KernelInterpolation", "single_point_interpolation", "constructor.jl"))


## StreamingInstability
include(joinpath(@__DIR__, "julia", "StreamingInstability", "StreamingInstability.jl"))
@reexport using .StreamingInstability
# include(joinpath(@__DIR__, "julia", "StreamingInstability", "growth_rate.jl"))



function _init_Core()      
    init_QR8buffer_bufferl!()        
end
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

function __init__()         
    _init_Core()     
end
end
