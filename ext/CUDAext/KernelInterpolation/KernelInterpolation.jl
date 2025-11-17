module KernelInterpolation
using PhantomRevealer
using CUDA

# Grid interpolation
include(joinpath(@__DIR__, "grid_interpolation", "grid_interpolation.jl"))

# Export function, marco, const...
for name in filter(s -> !startswith(string(s), "#"), names(@__MODULE__, all = true))
    if !startswith(String(name), "_") && (name != :eval) && (name != :include)
        @eval export $name
    end
end
end