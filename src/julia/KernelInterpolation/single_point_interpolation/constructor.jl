"""
Constructors for `InterpolationInput`
    by Wei-Shan Su,
    November 3, 2025

This file provides constructor overloads for building `InterpolationInput`
objects from a `PhantomRevealerDataFrame`.

All constructors return immutable, type-stable data containers for single-point SPH interpolation.
"""

import .KernelInterpolation: InterpolationInput

# Generate IntepolationInput
abstract type AbstractMassSource end
struct MassFromColumn <: AbstractMassSource
    name::Symbol
end
struct MassFromParams <: AbstractMassSource
    name::Symbol
end

"""
    InterpolationInput(data::PhantomRevealerDataFrame,
                       mass_from_column::MassFromColumn,
                       column_names::Vector{Symbol},
                       smoothed_kernel::Type{K}) where {K<:AbstractSPHKernel}

Construct a SPH interpolation container using **per-particle mass values** read from a specific column
in the `PhantomRevealerDataFrame`.

This variant of `InterpolationInput` assumes each particle has its own mass value stored in a data column.
It extracts the necessary SPH fields (positions, mass, density, smoothing length, and user-specified scalar fields),
promotes all numerical fields to a unified floating-point type, and returns an immutable container.

# Parameters
- `data::PhantomRevealerDataFrame`  
  Input particle dataset containing all physical quantities and parameters.
- `mass_from_column::MassFromColumn`  
  Specifies the column from which to read per-particle masses.  
- `column_names::Vector{Symbol}`  
  List of scalar fields (e.g., `[:P, :T]`) to interpolate, stored in the `quant` field.
- `smoothed_kernel::Type{K}`  
  Type of smoothing kernel to use, must subtype `AbstractSPHKernel`.

# Returns
- `::InterpolationInput{T, Vector{T}, K, NCOLUMN}`  
  Immutable, type-stable struct containing promoted particle data for interpolation.
# Notes
- All values are promoted to a consistent floating-point type `T`.
"""
function InterpolationInput(data::PhantomRevealerDataFrame, mass_from_column :: MassFromColumn, column_names::Vector{Symbol}, smoothed_kernel::Type{K}) where {K<:AbstractSPHKernel}
    N = get_npart(data)
    # Promote all to unified type
    Tprom = promote_type(
        eltype(data[!, :x]), eltype(data[!, :y]), eltype(data[!, :z]),
        eltype(data[!, :h]), eltype(data[!, :rho]), eltype(data[!, mass_from_column.name])
    )

    x = data[!, :x]
    y = data[!, :y]
    z = data[!, :z]
    h = data[!, :h]
    ρ = data[!, :rho]
    m = data[!, mass_from_column.name]

    # Build quant 
    NCOLUMN = length(column_names)
    quant = ntuple(i -> Vector{Tprom}(data[!, column_names[i]]), NCOLUMN)
    
    return InterpolationInput{Tprom, Vector{Tprom}, K, NCOLUMN}(
        N, smoothed_kernel, x, y, z, m, h, ρ, quant
    )
end


"""
    InterpolationInput(data::PhantomRevealerDataFrame,
                       mass_from_params::MassFromParams,
                       column_names::Vector{Symbol},
                       smoothed_kernel::Type{K}) where {K<:AbstractSPHKernel}

Construct a SPH interpolation container assuming **identical particle masses** read from the global
`params` dictionary in the `PhantomRevealerDataFrame`.

This variant of `InterpolationInput` is used when all particles share the same constant mass,
stored under a parameter key (e.g., `:mass`). It extracts all required SPH quantities,
promotes them to a unified floating-point type, and returns an immutable data container for single-point SPH interpolation.

# Parameters
- `data::PhantomRevealerDataFrame`  
  Input particle dataset containing all physical quantities and parameters.
- `mass_from_params::MassFromParams`  
  Specifies the parameter key used to obtain the constant particle mass from `data.params`.  
- `column_names::Vector{Symbol}`  
  List of scalar fields (e.g., `[:P, :T]`) to interpolate, stored in the `quant` field.
- `smoothed_kernel::Type{K}`  
  Type of smoothing kernel to use, must subtype `AbstractSPHKernel`.

# Returns
- `::InterpolationInput{T, Vector{T}, K, NCOLUMN}`  
  Immutable, type-stable struct containing promoted particle data for interpolation.
# Notes
- All values are promoted to a consistent floating-point type `T`.
"""
function InterpolationInput(data::PhantomRevealerDataFrame, mass_from_params :: MassFromParams, column_names::Vector{Symbol}, smoothed_kernel::Type{K}) where {K<:AbstractSPHKernel}
    N = get_npart(data)
    # Promote all to unified type
    Tprom = promote_type(
        eltype(data[!, :x]), eltype(data[!, :y]), eltype(data[!, :z]),
        eltype(data[!, :h]), eltype(data[!, :rho]), typeof(data.params[mass_from_params.name])
    )

    x = data[!, :x]
    y = data[!, :y]
    z = data[!, :z]
    h = data[!, :h]
    ρ = data[!, :rho]
    particle_mass = data.params[mass_from_params.name]
    m = fill(particle_mass, N)

    # Build quant 
    NCOLUMN = length(column_names)
    quant = ntuple(i -> Vector{Tprom}(data[!, column_names[i]]), NCOLUMN)
    
    return InterpolationInput{Tprom, Vector{Tprom}, K, NCOLUMN}(
        N, smoothed_kernel, x, y, z, m, h, ρ, quant
    )
end

export AbstractMassSource, MassFromColumn, MassFromParams, InterpolationInput