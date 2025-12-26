
abstract type AbstractExecutionBackend end 
struct CPUComputeBackend <: AbstractExecutionBackend end

"""
    initialize_interpolation(::CPUComputeBackend, grid_template::GeneralGrid{D}, input::ITPINPUT,
                         catalog::InterpolationCatalog{N, G, Div, C, L})

Initialises all CPU-side data structures required for grid-based SPH interpolation.
This method prepares the output grids, constructs the Linear Bounding Volume Hierarchy
(LBVH) for neighbour searches, and produces a concise catalog optimised for the
interpolation kernel.

# Parameters
- `::CPUComputeBackend`  
  Dispatch tag indicating that interpolation will be executed on the CPU.

- `grid_template::GeneralGrid{D}`  
  A template grid providing coordinate layout, dimensionality, and element types
  for all output grids.

- `input::ITPINPUT`  
  An `InterpolationInput` object containing particle data (positions, smoothing
  lengths, field values) and the SPH kernel used for interpolation.

- `catalog::InterpolationCatalog{N, G, Div, C, L}`  
  Full interpolation catalog containing symbolic quantity names and the mapping of
  scalar, gradient, divergence, and curl quantities.

# Returns
A tuple with:

1. `grids::NTuple{L, GeneralGrid{D}}`  
   The allocated output grids for each interpolated quantity.

2. `LBVH::LinearBVH`  
   The Linear Bounding Volume Hierarchy built from particle coordinates, used for
   fast neighbour searching.

3. `order::NTuple{L,Symbol}`  
   The ordered list of output quantity names.

4. `catalog_consice::InterpolationCatalogConcise`  
   A compact version of the catalog containing only slot and normalisation
   information and suitable for efficient CPU/GPU execution.

"""
function initialize_interpolation(:: CPUComputeBackend, grid_template::GeneralGrid{D}, input::ITPINPUT, catalog::InterpolationCatalog{N, G, Div, C, L}) where {D, N, G, Div, C, L, T <: AbstractFloat, ITPINPUT <: InterpolationInput{T}}
    # Generate a grid array for result
    order = catalog.ordered_names
    grids = ntuple(_ -> similar(grid_template), Val(L))

    # Generate a Linear BVH structure for neighborhood searching
    @info "Building LBVH..."
    @time begin
    LBVH = LinearBVH!(input, Val(D), CodeType = UInt64)
    end
    @info "End building LBVH..."

    # Consice catalog 
    catalog_consice = to_concise_catalog(catalog)

    return grids, LBVH, order, catalog_consice
end

@inline function _general_grid_interpolation_kernel!(:: CPUComputeBackend, grids :: NTuple{L, GeneralGrid{D}}, i :: Int, input :: ITPINPUT, catalog_consice :: InterpolationCatalogConcise{N, G, Div, C}, LBVH :: LinearBVH, itp_strategy::Type{ITPSTRATEGY} = itpSymmetric) where {D, N, G, Div, C, L, TF <: AbstractFloat, ITPINPUT <: InterpolationInput{TF}, ITPSTRATEGY <: AbstractInterpolationStrategy}
    # Get point
    point = grids[1].coor[i]
    
    # Particles searching
    ha = LBVH_find_nearest_h(LBVH, point)

    # Interpolation
    out_idx = 1        
    ## Scalar interpolation
    if N > 0
        scalars = quantities_interpolate(input, point, ha, LBVH, catalog_consice.scalar_slots, catalog_consice.scalar_snormalization, itp_strategy)
        @inbounds for j in 1:N
            grids[out_idx].grid[i] = scalars[j]
            out_idx += 1
        end
    end

    ## Gradients interpolation
    for j in 1:G
        slot = catalog_consice.grad_slots[j]
        grad_quant = slot == 0 ? gradient_density(input, point, ha, LBVH, itp_strategy) : gradient_quantity_interpolate(input, point, ha, LBVH, slot, itp_strategy)
        @inbounds for d in 1:D
            grids[out_idx].grid[i] = grad_quant[d]
            out_idx += 1
        end
    end

    ## Divergences interpolation
    for j in 1:Div
        slot = catalog_consice.div_slots[j]
        ax, ay, az = slot
        div_quant = divergence_quantity_interpolate(input, point, ha, LBVH, ax, ay, az, itp_strategy)
        grids[out_idx].grid[i] = div_quant
        out_idx += 1
    end

    ## Curls interpolation
    for j in 1:C
        slot = catalog_consice.curl_slots[j]
        ax, ay, az = slot
        curl_quant = curl_quantity_interpolate(input, point, ha, LBVH, ax, ay, az, itp_strategy)
        @inbounds for d in 1:D
            grids[out_idx].grid[i] = curl_quant[d]
            out_idx += 1
        end
    end
    @assert out_idx == L + 1 "InterpolationCatalog ordering mismatch: expected $L values, stored $(out_idx - 1)"
    return nothing
end

"""
    GeneralGrid_interpolation(backend::CPUComputeBackend, grid_template::GeneralGrid{D},
                          input::ITPINPUT, catalog::InterpolationCatalog{N, G, Div, C, L},
                          itp_strategy::Type{ITPSTRATEGY} = itpSymmetric)

Performs SPH interpolation over an arbitrary grid using CPU execution.
This routine dispatches to the CPU backend, prepares all interpolation structures,
and evaluates each grid point in parallel using threaded execution.

# Parameters
- `backend::CPUComputeBackend`  
  Execution backend specifying CPU-based interpolation.

- `grid_template::GeneralGrid{D}`  
  Template grid defining dimensionality, coordinate arrays, and memory layout of
  all output grids.

- `input::ITPINPUT`  
  The `InterpolationInput` holding particle positions, smoothing lengths, field
  data, and the SPH kernel.

- `catalog::InterpolationCatalog{N, G, Div, C, L}`  
  Full interpolation catalog describing which scalar, gradient, divergence, and
  curl quantities are to be produced.

- `itp_strategy::Type{ITPSTRATEGY}`  
  Interpolation strategy type controlling symmetric/gather/scatter modes.

# Returns
`GridInterpolationResult{L}` containing:
- `grids` — NTuple of output grids storing interpolated results.  
- `order` — Ordered list of all output quantity names, matching the grid tuple order.
"""
function GeneralGrid_interpolation(backend :: CPUComputeBackend, grid_template::GeneralGrid{D}, input::ITPINPUT, catalog::InterpolationCatalog{N, G, Div, C, L}, itp_strategy::Type{ITPSTRATEGY} = itpSymmetric) where {D, N, G, Div, C, L, T <: AbstractFloat, ITPINPUT <: InterpolationInput{T}, ITPSTRATEGY <: AbstractInterpolationStrategy}
    grids, LBVH, order, catalog_consice = initialize_interpolation(backend, grid_template, input, catalog)
    npoints = length(grid_template)
    @info"Start interpolation..."
    @time @inbounds @threads for i in 1:npoints
        # Do single point interpolation
        _general_grid_interpolation_kernel!(backend, grids, i, input, catalog_consice, LBVH, itp_strategy)

    end
    @info "End interpolation..."

    return GridInterpolationResult{L}(grids, order)
end