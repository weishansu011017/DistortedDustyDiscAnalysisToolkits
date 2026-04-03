"""
Constructors for `InterpolationInput`
    by Wei-Shan Su,
    November 3, 2025

This file provides core helpers for constructing `InterpolationInput` objects
from already-materialized particle columns.

Package-specific front-ends, such as `ParticleIO`, are expected to gather data
from their own container types and then call the internal helpers defined here.
"""

"""
    build_input(::CPUComputeBackend,
                x_col::AbstractVector,
                y_col::AbstractVector,
                z_col::AbstractVector,
                h_col::AbstractVector,
                ρ_col::AbstractVector,
                m_col::AbstractVector,
                quantity_columns::NTuple{NCOLUMN,<:AbstractVector};
                column_names::NTuple{NCOLUMN,Symbol},
                scalars::Tuple{Vararg{Symbol}}=(),
                gradients::Tuple{Vararg{Symbol}}=(),
                divergences::Tuple{Vararg{Symbol}}=(),
                curls::Tuple{Vararg{Symbol}}=(),
                smoothed_kernel::Type{K}=M5_spline) where {K<:AbstractSPHKernel,NCOLUMN}

Construct a CPU-side `InterpolationInput` and its corresponding
`InterpolationCatalog` from already-materialized particle columns.

This helper promotes all input columns to a common element type, materializes
them as dense CPU `Vector`s, constructs an `InterpolationInput`, and then
builds a 3D `InterpolationCatalog` from the provided extra quantity names.
The catalog request names are resolved only against `column_names`, which must
correspond one-to-one with `quantity_columns`.

Base interpolation fields `x`, `y`, `z`, `h`, `ρ`, and `m` are supplied
explicitly through positional arguments and are not part of the catalog lookup
namespace.

# Parameters
- `::CPUComputeBackend`: Compute backend selector for the CPU interpolation
  path.
- `x_col::AbstractVector`: Particle `x` coordinates.
- `y_col::AbstractVector`: Particle `y` coordinates.
- `z_col::AbstractVector`: Particle `z` coordinates.
- `h_col::AbstractVector`: Particle smoothing lengths.
- `ρ_col::AbstractVector`: Particle densities.
- `m_col::AbstractVector`: Particle masses.
- `quantity_columns::NTuple{NCOLUMN,<:AbstractVector}`: Extra particle
  quantity columns to be attached to the interpolation input.

# Keyword Arguments
| Keyword | Type | Default | Description |
|---|---|---|---|
| `column_names` | `NTuple{NCOLUMN,Symbol}` | — | Names associated one-to-one with `quantity_columns`. These names define the catalog lookup namespace. |
| `scalars` | `Tuple{Vararg{Symbol}}` | `()` | Names of extra scalar quantities to interpolate directly. |
| `gradients` | `Tuple{Vararg{Symbol}}` | `()` | Names of extra scalar quantities whose spatial gradients should be computed. |
| `divergences` | `Tuple{Vararg{Symbol}}` | `()` | Base names of extra vector quantities whose divergences should be computed. |
| `curls` | `Tuple{Vararg{Symbol}}` | `()` | Base names of extra vector quantities whose curls should be computed. |
| `smoothed_kernel` | `Type{K}` where `K<:AbstractSPHKernel` | `M5_spline` | SPH kernel type used when constructing the `InterpolationInput`. |

# Returns
- `Tuple{InterpolationInput,InterpolationCatalog{3,N,G,Div,C,L}}`: A pair
  consisting of the materialized CPU interpolation input and the corresponding
  3D interpolation catalog for the requested extra quantities.
"""
function build_input(:: CPUComputeBackend,
    x_col :: AbstractVector, y_col :: AbstractVector, z_col :: AbstractVector, h_col :: AbstractVector, ρ_col :: AbstractVector, m_col :: AbstractVector, quantity_columns :: NTuple{NCOLUMN, <: AbstractVector};
    column_names :: NTuple{NCOLUMN, Symbol}, 
    scalars :: Tuple{Vararg{Symbol}} = (), gradients :: Tuple{Vararg{Symbol}} = (), divergences :: Tuple{Vararg{Symbol}} = (), curls :: Tuple{Vararg{Symbol}} = (), 
    smoothed_kernel::Type{K} = M5_spline) where {K <: AbstractSPHKernel, NCOLUMN}

    # Get promoted type for all columns to ensure consistent numeric types in InterpolationInput
    Tprom = if isempty(quantity_columns)
        promote_type(
            eltype(x_col),
            eltype(y_col),
            eltype(z_col),
            eltype(h_col),
            eltype(ρ_col),
            eltype(m_col),
        )
    else
        promote_type(
            eltype(x_col),
            eltype(y_col),
            eltype(z_col),
            eltype(h_col),
            eltype(ρ_col),
            eltype(m_col),
            (eltype(column) for column in quantity_columns)...,
        )
    end

    x = Vector{Tprom}(x_col)
    y = Vector{Tprom}(y_col)
    z = Vector{Tprom}(z_col)
    h = Vector{Tprom}(h_col)
    ρ = Vector{Tprom}(ρ_col)
    m = Vector{Tprom}(m_col)

    quant = ntuple(i -> Vector{Tprom}(quantity_columns[i]), NCOLUMN)

    input = InterpolationInput(
        x,
        y,
        z,
        m,
        h,
        ρ,
        quant;
        smoothed_kernel = smoothed_kernel,
    )

    catalog = InterpolationCatalog(
        column_names, Val(3);
        scalars = scalars,
        gradients = gradients,
        divergences = divergences,
        curls = curls,
    )

    return input, catalog
end
