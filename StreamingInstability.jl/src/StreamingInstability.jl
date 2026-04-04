"""
StreamingInstability

Implements analytical and numerical routines for estimating the linear
growth rate of the streaming instability following Chen & Lin (2021).
This module provides core functions to evaluate the dispersion relation
and characterize the stability of dust-gas mixtures in protoplanetary discs.

Currently implemented:
- Growth-rate calculation based on linear perturbation analysis.
"""
module StreamingInstability

using Logging
using Statistics
using LinearAlgebra
using StaticArrays
using TinyEigvals

# Growth rate estimation through Chen & Lin (2021)
include(joinpath(@__DIR__, "julia", "classical_SI_growth_rate.jl"))

# Package metadata helpers.
version() = pkgversion(@__MODULE__)

function about()
    @info "StreamingInstability Module\n  Version: $(version())\n  Made by Wei-Shan Su, Apr 2026"
    return nothing
end

# Export function, macro, const...
for name in filter(s -> !startswith(string(s), "#"), names(@__MODULE__, all = true))
    if !startswith(String(name), "_") && (name != :eval) && (name != :include)
        @eval export $name
    end
end
end
