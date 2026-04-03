"""
    LineSamples_interpolation(backend::CPUComputeBackend, grid_template::LineSamples{D,TF},
                              input::InterpolationInput{3,TF}, catalog::InterpolationCatalog{3, N, G, Div, C, L},
                              itp_strategy::Type{ITPSTRATEGY}=itpScatter)

Perform CPU-based SPH interpolation for an unstructured collection of line samples.

This routine initialises the interpolation data structures for the CPU backend,
allocates the output `LineSamples` containers, and evaluates each line sample in
parallel using threaded execution.

Each sample is interpreted as a line primitive defined by the corresponding
origin and direction stored in `grid_template`. For the `i`-th sample, the
interpolation kernel evaluates line-integrated quantities associated with that
line and stores the resulting scalar values into the output grids.

At present, this routine only supports `itpScatter`. For line-integrated
samples there is no well-defined query smoothing length `ha`, so
`itpGather` and `itpSymmetric` are rejected explicitly.

# Parameters
- `backend::CPUComputeBackend`
  Execution backend specifying CPU-based interpolation.

- `grid_template::LineSamples{D,TF}`
  Template sample container defining the dimensionality, line geometry, and
  output container layout.

- `input::InterpolationInput{3,TF}`
  An `InterpolationInput` object containing particle positions, smoothing
  lengths, field values, and the SPH kernel.

- `catalog::InterpolationCatalog{3, N, G, Div, C, L}`
  Interpolation catalog describing the requested output quantities.
  In the current implementation, only scalar line-integrated quantities are
  supported.

- `itp_strategy::Type{ITPSTRATEGY}`
  Interpolation strategy type. Only `itpScatter` is supported.

# Returns
- `GridBundle`
  A bundle containing:
  - `grids` : output `LineSamples` containers storing the interpolated scalar values
  - `names` : ordered quantity names matching the output grid order
"""
function LineSamples_interpolation(backend :: CPUComputeBackend, grid_template :: LineSamples{3, TF}, input :: InterpolationInput{3, TF}, catalog :: InterpolationCatalog{3, N, 0, 0, 0, N}, itp_strategy :: Type{ITPSTRATEGY} = itpScatter) where {N, TF <: AbstractFloat, ITPSTRATEGY <: AbstractInterpolationStrategy}
    itp_strategy === itpScatter || throw(ArgumentError(
        "LineSamples_interpolation only supports itpScatter. " *
        "Line-integrated samples do not have a well-defined query smoothing length ha, " *
        "so itpGather and itpSymmetric are not supported."
    ))

    grids_result, LBVH, names, catalog_consice = initialize_interpolation(backend, grid_template, input, catalog)
    npoints = length(grid_template)

    @info "     SPH Interpolation: Start interpolation..."
    @inbounds @threads for i in 1:npoints
        _line_samples_interpolation_kernel!(backend, grids_result, i, input, catalog_consice, LBVH, itp_strategy)
    end
    @info "     SPH Interpolation: End interpolation..."

    grids = grids_result
    return GridBundle(grids, names)
end


# Interpolation kernels
## Line-integrated samples use particle-side smoothing lengths only
@inline function _line_samples_interpolation_kernel!(:: CPUComputeBackend, grids::NTuple{N, LineSamples{3, TF}}, i::Int, input::InterpolationInput{3, TF}, catalog_consice::InterpolationCatalogConcise{3, N, 0, 0, 0}, LBVH::LinearBVH, ::Type{itpScatter}) where {N, TF <: AbstractFloat}
    # Get line sample geometry
    @inbounds begin
        geometry = grids[1]

        xoa = geometry.origin[1][i]
        yoa = geometry.origin[2][i]
        zoa = geometry.origin[3][i]
        origin::NTuple{3, TF} = (xoa, yoa, zoa)

        xda = geometry.direction[1][i]
        yda = geometry.direction[2][i]
        zda = geometry.direction[3][i]
        direction::NTuple{3, TF} = (xda, yda, zda)
    end

    # Interpolation
    scalar_slots::NTuple{N, Int} = catalog_consice.scalar_slots
    scalar_snormalization::NTuple{N, Bool} = catalog_consice.scalar_snormalization
    scalars::NTuple{N, TF} = _line_integrated_quantities_interpolate_kernel(input, origin, direction, LBVH, scalar_slots, scalar_snormalization, itpScatter)

    # Store results
    if N > 0
        @inbounds for j in 1:N
            grids[j].grid[i] = scalars[j]
        end
    end

    return nothing
end
