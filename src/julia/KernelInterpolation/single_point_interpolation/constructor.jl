"""
Constructors for `InterpolationInput`
    by Wei-Shan Su,
    November 3, 2025

This file provides high-level builders for `InterpolationInput` objects derived
from a `PhantomRevealerDataFrame`.

All constructors return immutable, type-stable data containers for single-point SPH interpolation.
"""

import .KernelInterpolation: InterpolationCatalog, InterpolationInput

# For field suffixes
const _XYZ_SUFFIXES = (:x, :y, :z)
@inline function _vector_components(name::Symbol; suffixes=_XYZ_SUFFIXES)::NTuple{3,Symbol}
  return ntuple(i -> Symbol(name, suffixes[i]), 3)
end

# Generate IntepolationInput
abstract type AbstractMassSource end
struct MassFromColumn <: AbstractMassSource
    name::Symbol
end
struct MassFromParams <: AbstractMassSource
    name::Symbol
end

"""
  build_input(data::PhantomRevealerDataFrame,
                       mass_from_column::MassFromColumn;
                       scalars::Vector{Symbol},
                       gradients::Vector{Symbol}=Symbol[],
                       divergences::Vector{Symbol}=Symbol[],
                       curls::Vector{Symbol}=Symbol[],
                       smoothed_kernel::Type{K}=M5_spline) where {K<:AbstractSPHKernel}

Construct a SPH interpolation container using **per-particle mass values** read from a specific column
in the `PhantomRevealerDataFrame`.

This variant of `InterpolationInput` assumes each particle has its own mass value stored in a data column.
It extracts the necessary SPH fields (positions, mass, density, smoothing length, and user-requested scalar/vector
components), promotes all numerical fields to a unified floating-point type, and returns a pair containing the
immutable container and a host-side catalog for name-to-column lookup.

# Parameters
- `data::PhantomRevealerDataFrame`  
  Input particle dataset containing all physical quantities and parameters.
- `mass_from_column::MassFromColumn`  
  Specifies the column from which to read per-particle masses.  
- `scalars::Vector{Symbol}`  
  Scalar fields to materialise inside `quant` and expose for direct interpolation.  
- `gradients::Vector{Symbol}`  
  Scalar fields whose gradients will be requested downstream. Ensures the columns exist in `quant`.  
- `divergences::Vector{Symbol}`  
  Vector quantities whose `(x,y,z)` components should be present for divergence kernels.  
- `curls::Vector{Symbol}`  
  Vector quantities whose `(x,y,z)` components should be present for curl kernels.  
- `smoothed_kernel::Type{K}`  
  Type of smoothing kernel to use, must subtype `AbstractSPHKernel`.

# Returns
- `(::InterpolationInput{T, Vector{T}, K, NCOLUMN}, ::InterpolationCatalog)`  
  Immutable particle container together with a lookup catalog that maps the declared symbols to column indices.
# Notes
- All values are promoted to a consistent floating-point type `T`.
"""
function build_input(data::PhantomRevealerDataFrame,
                            mass_from_column::MassFromColumn;
                            scalars::Vector{Symbol},
                            gradients::Vector{Symbol}=Symbol[],
                            divergences::Vector{Symbol}=Symbol[],
                            curls::Vector{Symbol}=Symbol[],
                            smoothed_kernel::Type{K}=M5_spline) where {K<:AbstractSPHKernel}
    N = get_npart(data)

    # Collect every column we must materialise into the quant tuple
    column_names = Symbol[]
    for name in scalars
        name in column_names || push!(column_names, name)
    end
    for name in gradients
        name in column_names || push!(column_names, name)
    end
    for name in divergences
        for comp in _vector_components(name)
            comp in column_names || push!(column_names, comp)
        end
    end
    for name in curls
        for comp in _vector_components(name)
            comp in column_names || push!(column_names, comp)
        end
    end

    available_cols = Set(Symbol.(names(data)))
    missing_columns = Symbol[]
    for base in (:x, :y, :z, :h, :rho, mass_from_column.name)
    if !(base in available_cols) && !(base in missing_columns)
      push!(missing_columns, base)
        end
    end
    for name in column_names
    if !(name in available_cols) && !(name in missing_columns)
      push!(missing_columns, name)
        end
    end
    if !isempty(missing_columns)
    missing_list = join(string.(missing_columns), ", ")
        throw(ArgumentError("Missing columns in PhantomRevealerDataFrame: " * missing_list))
    end

    x_col = data[!, :x]
    y_col = data[!, :y]
    z_col = data[!, :z]
    h_col = data[!, :h]
    ρ_col = data[!, :rho]
    m_col = data[!, mass_from_column.name]

    # Promote all to unified type
    Tprom = if isempty(column_names)
        promote_type(eltype(x_col), eltype(y_col), eltype(z_col), eltype(h_col), eltype(ρ_col), eltype(m_col))
    else
        promote_type(
            eltype(x_col), eltype(y_col), eltype(z_col), eltype(h_col), eltype(ρ_col), eltype(m_col),
            (eltype(data[!, name]) for name in column_names)...
        )
    end

    x = Vector{Tprom}(x_col)
    y = Vector{Tprom}(y_col)
    z = Vector{Tprom}(z_col)
    h = Vector{Tprom}(h_col)
    ρ = Vector{Tprom}(ρ_col)
    m = Vector{Tprom}(m_col)

    NCOLUMN = length(column_names)
    quant = ntuple(i -> Vector{Tprom}(data[!, column_names[i]]), NCOLUMN)

    input = InterpolationInput{Tprom, Vector{Tprom}, K, NCOLUMN}(
        N, smoothed_kernel(), x, y, z, m, h, ρ, quant
    )

    column_index = Dict{Symbol,Int}(name => idx for (idx, name) in enumerate(column_names))

    scalar_names = Tuple(scalars)
    scalar_slots = Tuple(column_index[name] for name in scalars)
    grad_names   = Tuple(gradients)
    grad_slots   = Tuple(column_index[name] for name in gradients)
    div_names    = Tuple(divergences)
    div_slots    = Tuple(begin
        comps = _vector_components(name)
        ntuple(i -> column_index[comps[i]], 3)
    end for name in divergences)
    curl_names   = Tuple(curls)
    curl_slots   = Tuple(begin
        comps = _vector_components(name)
        ntuple(i -> column_index[comps[i]], 3)
    end for name in curls)

    catalog = InterpolationCatalog(
        scalar_names, scalar_slots,
        grad_names, grad_slots,
        div_names, div_slots,
        curl_names, curl_slots
    )

    return input, catalog
end

"""
  build_input(data::PhantomRevealerDataFrame,
             mass_from_params::MassFromParams;
             scalars::Vector{Symbol},
             gradients::Vector{Symbol}=Symbol[],
             divergences::Vector{Symbol}=Symbol[],
             curls::Vector{Symbol}=Symbol[],
             smoothed_kernel::Type{K}=M5_spline) where {K<:AbstractSPHKernel}

Construct a SPH interpolation container using a **constant particle mass** stored in
`data.params` under the provided key. The per-particle mass column in the data frame
is ignored, allowing datasets without a dedicated mass field.

This builder mirrors the column-based version: it extracts the required SPH fields
(positions, density, smoothing length, and any user-requested scalar/vector
quantities), promotes them to a unified floating-point type, and returns both the
immutable container and a host-side catalog for name-to-column lookup.

# Parameters
- `data::PhantomRevealerDataFrame`  
  Input particle dataset containing all physical quantities and parameters.
- `mass_from_params::MassFromParams`  
  Parameter descriptor from which the constant particle mass is retrieved.  
- `scalars::Vector{Symbol}`  
  Scalar fields to materialise inside `quant` and expose for direct interpolation.  
- `gradients::Vector{Symbol}`  
  Scalar fields whose gradients will be requested downstream. Ensures the columns exist in `quant`.  
- `divergences::Vector{Symbol}`  
  Vector quantities whose `(x,y,z)` components should be present for divergence kernels.  
- `curls::Vector{Symbol}`  
  Vector quantities whose `(x,y,z)` components should be present for curl kernels.  
- `smoothed_kernel::Type{K}`  
  Type of smoothing kernel to use, must subtype `AbstractSPHKernel`.

# Returns
- `(::InterpolationInput{T, Vector{T}, K, NCOLUMN}, ::InterpolationCatalog)`  
  Immutable particle container together with a lookup catalog that maps the declared symbols to column indices.
# Notes
- All values are promoted to a consistent floating-point type `T`.
"""
function build_input(data::PhantomRevealerDataFrame,
                            mass_from_params::MassFromParams;
                            scalars::Vector{Symbol},
                            gradients::Vector{Symbol}=Symbol[],
                            divergences::Vector{Symbol}=Symbol[],
                            curls::Vector{Symbol}=Symbol[],
                            smoothed_kernel::Type{K}=M5_spline) where {K<:AbstractSPHKernel}
    N = get_npart(data)

    # Collect every column we must materialise into the quant tuple
    column_names = Symbol[]
    for name in scalars
        name in column_names || push!(column_names, name)
    end
    for name in gradients
        name in column_names || push!(column_names, name)
    end
    for name in divergences
        for comp in _vector_components(name)
            comp in column_names || push!(column_names, comp)
        end
    end
    for name in curls
        for comp in _vector_components(name)
            comp in column_names || push!(column_names, comp)
        end
    end

    available_cols = Set(Symbol.(names(data)))
    missing_columns = Symbol[]
    for base in (:x, :y, :z, :h, :rho)
    if !(base in available_cols) && !(base in missing_columns)
      push!(missing_columns, base)
        end
    end
    for name in column_names
    if !(name in available_cols) && !(name in missing_columns)
      push!(missing_columns, name)
        end
    end
    if !isempty(missing_columns)
    missing_list = join(string.(missing_columns), ", ")
        throw(ArgumentError("Missing columns in PhantomRevealerDataFrame: " * missing_list))
    end

    x_col = data[!, :x]
    y_col = data[!, :y]
    z_col = data[!, :z]
    h_col = data[!, :h]
    ρ_col = data[!, :rho]

    # Promote all to unified type
    Tprom = if isempty(column_names)
        promote_type(eltype(x_col), eltype(y_col), eltype(z_col), eltype(h_col), eltype(ρ_col))
    else
        promote_type(
            eltype(x_col), eltype(y_col), eltype(z_col), eltype(h_col), eltype(ρ_col),
            (eltype(data[!, name]) for name in column_names)...
        )
    end

    x = Vector{Tprom}(x_col)
    y = Vector{Tprom}(y_col)
    z = Vector{Tprom}(z_col)
    h = Vector{Tprom}(h_col)
    ρ = Vector{Tprom}(ρ_col)
    particle_mass = data.params[mass_from_params.name]
    m = Tprom.(fill(particle_mass, N))

    NCOLUMN = length(column_names)
    quant = ntuple(i -> Vector{Tprom}(data[!, column_names[i]]), NCOLUMN)

    input = InterpolationInput{Tprom, Vector{Tprom}, K, NCOLUMN}(
        N, (smoothed_kernel()), x, y, z, m, h, ρ, quant
    )

    column_index = Dict{Symbol,Int}(name => idx for (idx, name) in enumerate(column_names))

    scalar_names = Tuple(scalars)
    scalar_slots = Tuple(column_index[name] for name in scalars)
    grad_names   = Tuple(gradients)
    grad_slots   = Tuple(column_index[name] for name in gradients)
    div_names    = Tuple(divergences)
    div_slots    = Tuple(begin
        comps = _vector_components(name)
        ntuple(i -> column_index[comps[i]], 3)
    end for name in divergences)
    curl_names   = Tuple(curls)
    curl_slots   = Tuple(begin
        comps = _vector_components(name)
        ntuple(i -> column_index[comps[i]], 3)
    end for name in curls)

    catalog = InterpolationCatalog(
        scalar_names, scalar_slots,
        grad_names, grad_slots,
        div_names, div_slots,
        curl_names, curl_slots
    )

    return input, catalog
end

export AbstractMassSource, MassFromColumn, MassFromParams, InterpolationInput, build_input