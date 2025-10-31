"""
Tools

A collection of general-purpose utilities for numerical analysis and data handling.
This module provides functions commonly used across PhantomRevealer,
including:

- Thermodynamic and equation-of-state properties (`eos_properties.jl`)
- Logging and message control (`logging.jl`)
- Coordinate transformation utilities (`coordinate_transformations.jl`)
- Array and matrix operations (`array_operations.jl`)

These functions serve as lightweight building blocks for higher-level
modules and analyses.
"""
module Tools
using .Threads 
using Statistics
using LinearAlgebra
using Logging


# Tools
include(joinpath(@__DIR__, "eos_properties.jl"))
include(joinpath(@__DIR__, "logging.jl"))
include(joinpath(@__DIR__, "coordinate_transformations.jl"))
include(joinpath(@__DIR__, "array_operations.jl"))


# Export function, marco, const...
for name in filter(s -> !startswith(string(s), "#"), names(@__MODULE__, all = true))
    if !startswith(String(name), "_") && (name != :eval) && (name != :include)
        @eval export $name
    end
end
end