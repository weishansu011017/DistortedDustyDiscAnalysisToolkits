module KernelInterpolation
using Partia
using CUDA

# Line-integrated interpolation
include(joinpath(@__DIR__, "line_integrated_interpolation", "kernels", "line_integrated_scalar_interpolation.jl"))

# Grid interpolation
include(joinpath(@__DIR__, "grid_interpolation", "PointSamples_interpolation.jl"))
include(joinpath(@__DIR__, "grid_interpolation", "LineSamples_interpolation.jl"))

# Export function, marco, const...
for name in filter(s -> !startswith(string(s), "#"), names(@__MODULE__, all = true))
    if !startswith(String(name), "_") && (name != :eval) && (name != :include)
        @eval export $name
    end
end
end
