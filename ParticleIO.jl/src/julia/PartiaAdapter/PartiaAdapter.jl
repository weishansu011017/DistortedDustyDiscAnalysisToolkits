"""
PartiaAdapter

Adapters that convert `ParticleIO` data structures into `Partia`
interpolation inputs.

This module owns the `ParticleDataFrame`-specific convenience API that bridges
particle-side tabular data to `Partia.KernelInterpolation`.
"""
module PartiaAdapter

using ParticleIO.Particles
using Partia.KernelInterpolation: AbstractSPHKernel, M5_spline

include(joinpath(@__DIR__, "MassSource.jl"))
include(joinpath(@__DIR__, "build_input.jl"))

# Export function, macro, const...
for name in filter(s -> !startswith(string(s), "#"), names(@__MODULE__, all = true))
    if !startswith(String(name), "_") && (name != :eval) && (name != :include)
        @eval export $name
    end
end
end
