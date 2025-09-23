module PhantomRevealer
# Include the Julia Module

using .Threads 
using Pkg
using Logging
using Statistics
using LinearAlgebra
using StaticArrays
using DataFrames
# using Interpolations
##################### Core #####################
## Tools
include(joinpath(@__DIR__, "julia", "tools", "eos_properties.jl"))
include(joinpath(@__DIR__, "julia", "tools", "logging.jl"))
include(joinpath(@__DIR__, "julia", "tools", "coordinate_transformations.jl"))

## KernelInterpolation
include(joinpath(@__DIR__, "julia", "KernelInterpolation", "table", "los_tables.jl"))
include(joinpath(@__DIR__, "julia", "KernelInterpolation", "kernel_function.jl"))
include(joinpath(@__DIR__, "julia", "KernelInterpolation", "grid.jl"))

## StreamingInstability
include(joinpath(@__DIR__, "julia", "StreamingInstability", "growth_rate.jl"))


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


# Export function, marco, const...
for name in filter(s -> !startswith(string(s), "#"), names(@__MODULE__, all = true))
    if !startswith(String(name), "_") && (name != :eval) && (name != :include)
        @eval export $name
    end
end
end
