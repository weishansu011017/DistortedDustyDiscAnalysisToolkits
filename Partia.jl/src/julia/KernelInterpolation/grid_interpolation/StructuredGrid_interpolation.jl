"""
    StructuredGrid_interpolation(backend::B,
                                 ::Type{COORD},
                                 grid_template::StructuredGrid{3},
                                 input::InterpolationInput{3,T},
                                 catalog::InterpolationCatalog{3,N,G,Div,C,L},
                                 itp_strategy::Type{ITPSTRATEGY}=itpSymmetric) where
                                 {COORD,N,G,Div,C,L,
                                  T<:AbstractFloat,
                                  ITPSTRATEGY<:AbstractInterpolationStrategy,
                                  B<:AbstractExecutionBackend}

Perform SPH interpolation on a structured grid and return interpolated fields as a `GridBundle`.

This routine converts a `StructuredGrid` template into a flattened `PointSamples` representation,
dispatches interpolation to `PointSamples_interpolation` on the specified execution backend,
then restores each interpolated field back to `StructuredGrid` layout using the copied axes.

# Parameters
- `backend::B`:
  Execution backend used by `PointSamples_interpolation` (e.g. CPU/CUDA/Metal backends).

- `::Type{COORD}`:
  Explicit coordinate-system dispatch controlling how the structured-grid axes
  are interpreted before they are flattened into Cartesian sample coordinates.

- `grid_template::StructuredGrid{3}`:
  Structured grid template providing the coordinate axes and logical grid shape.

- `input::InterpolationInput{3,T}`:
  Interpolation input containing particle coordinates, smoothing lengths, field data, and the SPH kernel.

- `catalog::InterpolationCatalog{3,N,G,Div,C,L}`:
  Interpolation catalog describing which scalar, gradient, divergence, and curl quantities to compute.
  The number of output grids is `L`.

- `itp_strategy::Type{ITPSTRATEGY}=itpSymmetric`:
  Interpolation strategy controlling symmetric/gather/scatter modes.

# Returns
- `GridBundle{L,<:StructuredGrid}`:
  A bundle containing:
  - `grids`: `NTuple{L,StructuredGrid{D,T}}` storing interpolated results for each requested quantity.
  - `names`: `NTuple{L,Symbol}` giving the corresponding quantity names in the same order.
"""
function StructuredGrid_interpolation(backend :: B, ::Type{COORD}, grid_template::StructuredGrid{3}, input::InterpolationInput{3, T}, catalog::InterpolationCatalog{3, N, G, Div, C, L}, itp_strategy::Type{ITPSTRATEGY} = itpSymmetric) where {COORD <: AbstractCoordinateSystem, N, G, Div, C, L, T <: AbstractFloat, ITPSTRATEGY <: AbstractInterpolationStrategy, B <: AbstractExecutionBackend}
    @info "     SPH Interpolation: Flatterning grid..."
    flatten_grid = flatten(COORD, grid_template)
    @info "     SPH Interpolation: End flatterning grid."
    generalgrid_bundle = PointSamples_interpolation(backend, flatten_grid, input, catalog, itp_strategy)
    names = generalgrid_bundle.names

    @info "     SPH Interpolation: Restoring grid..."
    newT = datatype(generalgrid_bundle)
    axes = ntuple(d -> begin
        ax  = grid_template.axes[d]
        out = similar(ax, newT)
        map!(newT, out, ax)           
        out
    end, Val(3))
    structuredgrids = ntuple(i -> restore_struct(COORD, generalgrid_bundle.grids[i], axes), Val(L))
    @info "     SPH Interpolation: End restoring grid."
    return GridBundle(structuredgrids, names)
end
