module PhantomRevealer
# Include the Julia Module
# With the order of level
_module_location = @__DIR__

using Statistics
using LinearAlgebra
using StaticArrays

# Core
include("$_module_location/table/los_tables.jl")
include("$_module_location/julia/kernel_function.jl")

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


# Export function, marco, const...
for name in filter(s -> !startswith(string(s), "#"), names(@__MODULE__, all = true))
    if !startswith(String(name), "_") && (name != :eval) && (name != :include)
        @eval export $name
    end
end
end
