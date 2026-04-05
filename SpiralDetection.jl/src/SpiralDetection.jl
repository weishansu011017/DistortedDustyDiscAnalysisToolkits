module SpiralDetection

using Logging
using Statistics

# include(joinpath(@__DIR__, "julia", "spiral_detection", "logspiral_detection.jl"))

# Package metadata helpers.
version() = pkgversion(@__MODULE__)

function about()
    @info "SpiralDetection Module\n  Version: $(version())\n  Made by Wei-Shan Su, Apr 2026"
    return nothing
end

# Export function, macro, const...
for name in filter(s -> !startswith(string(s), "#"), names(@__MODULE__, all = true))
    if !startswith(String(name), "_") && (name != :eval) && (name != :include)
        @eval export $name
    end
end

end
