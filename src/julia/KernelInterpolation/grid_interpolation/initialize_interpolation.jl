"""
    initialize_interpolation(::CPUComputeBackend, grid_template::AbstractSamples{D,T},
                             input::ITPINPUT, catalog::InterpolationCatalog{N, G, Div, C, L})

Initialise all CPU-side data structures required for sample-based SPH interpolation.

This method allocates one output sample container for each requested interpolated
quantity, constructs a Linear Bounding Volume Hierarchy (LBVH) for neighbour
searches, and produces a concise interpolation catalog suitable for efficient
kernel execution.

The method applies to unstructured sample containers derived from `AbstractSamples`,
such as `PointSamples` and `LineSamples`, provided that `similar(grid_template)`
returns a compatible output container.

# Parameters
- `::CPUComputeBackend`
  Dispatch tag indicating that interpolation will be executed on the CPU.

- `grid_template::AbstractSamples{D,T}`
  A template sample container providing dimensionality, element type, and output
  container layout for all interpolated quantities.

- `input::ITPINPUT`
  An `InterpolationInput` object containing particle data (positions, smoothing
  lengths, field values) and the SPH kernel used for interpolation.

- `catalog::InterpolationCatalog{N, G, Div, C, L}`
  Full interpolation catalog containing symbolic quantity names and the mapping of
  scalar, gradient, divergence, and curl quantities.

# Returns
A tuple with:

1. `grids::NTuple{L,<:AbstractSamples{D,T}}`
   The allocated output sample containers for each interpolated quantity.

2. `LBVH::LinearBVH`
   The Linear Bounding Volume Hierarchy built from particle coordinates, used for
   fast neighbour searching.

3. `order::NTuple{L,Symbol}`
   The ordered list of output quantity names.

4. `catalog_consice::InterpolationCatalogConcise`
   A compact version of the catalog containing only slot and normalization
   information and suitable for efficient CPU/GPU execution.
"""
function initialize_interpolation(:: CPUComputeBackend, grid_template :: AS, input :: ITPINPUT, catalog :: InterpolationCatalog{N, G, Div, C, L}) where {D, N, G, Div, C, L, T <: AbstractFloat, ITPINPUT <: InterpolationInput{T}, AS <: AbstractSamples{D, T}}
    # Generate a grid array for result
    @info "     SPH Interpolation: Allocating output grids..."
    names = catalog.ordered_names
    grids = ntuple(_ -> similar(grid_template), Val(L))
    @info "     SPH Interpolation: End allocating output grids..."

    # Generate a Linear BVH structure for neighborhood searching
    @info "     SPH Interpolation: Building LBVH..."
    LBVH = LinearBVH!(input, Val(D), CodeType = UInt64)
    @info "     SPH Interpolation: End building LBVH..."

    # Consice catalog 
    catalog_consice = to_concise_catalog(catalog)

    return grids, LBVH, names, catalog_consice
end
