module PhantomRevealer
# Include the Julia Module
# With the order of level
_module_location = @__DIR__
#Level 1 (Package and information)
include("$_module_location/julia/module_initialization.jl")
include("$_module_location/julia/logging.jl")
include("$_module_location/julia/growth_rate.jl")
#Level 2 (SPH Mathematics)
include("$_module_location/julia/mathematical_tools.jl")
include("$_module_location/table/los_tables.jl")
include("$_module_location/julia/kernel_function.jl")
include("$_module_location/julia/eos_properties.jl")
#Level 3 (Data Structure)
include("$_module_location/julia/grid.jl")
include("$_module_location/julia/PhantomRevealerDataFrame.jl")
#Level 4 (Sigal point analysis and File read)
include("$_module_location/julia/physical_quantity.jl")
include("$_module_location/julia/read_phantom.jl")
#Level 5 (Analysis)
include("$_module_location/julia/grid_interpolation.jl")
include("$_module_location/julia/ridge_detection.jl")
include("$_module_location/julia/Hough_transform.jl")
include("$_module_location/julia/beam_search.jl")
#Level 6 (Extract data)
include("$_module_location/julia/Makie_backend.jl")
include("$_module_location/julia/extract_data.jl")
include("$_module_location/julia/result_toolkits.jl")
# include("$_module_location/julia/spiral_detection.jl")


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
    n = nthreads() 
    init_growth_buffers!()                          
    KDSearching_scratch[] = [sizehint!(Int[], 1024) for _ = 1:n]
end

# Export function, marco, const...
for name in filter(s -> !startswith(string(s), "#"), names(@__MODULE__, all = true))
    if !startswith(String(name), "_") && (name != :eval) && (name != :include)
        @eval export $name
    end
end
end
