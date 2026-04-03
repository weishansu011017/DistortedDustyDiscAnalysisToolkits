"""
    PointSamples_interpolation(backend::CPUComputeBackend, grid_template::PointSamples{D},
                          input::InterpolationInput{3,TF}, catalog::InterpolationCatalog{3, N, G, Div, C, L},
                          itp_strategy::Type{ITPSTRATEGY} = itpSymmetric)

Performs SPH interpolation over an arbitrary grid using CPU execution.
This routine dispatches to the CPU backend, prepares all interpolation structures,
and evaluates each grid point in parallel using threaded execution.

# Parameters
- `backend::CPUComputeBackend`  
  Execution backend specifying CPU-based interpolation.

- `grid_template::PointSamples{D}`  
  Template grid defining dimensionality, coordinate arrays, and memory layout of
  all output grids.

- `input::InterpolationInput{3,TF}`  
  The `InterpolationInput` holding particle positions, smoothing lengths, field
  data, and the SPH kernel.

- `catalog::InterpolationCatalog{3, N, G, Div, C, L}`  
  Full interpolation catalog describing which scalar, gradient, divergence, and
  curl quantities are to be produced.

- `itp_strategy::Type{ITPSTRATEGY}`  
  Interpolation strategy type controlling symmetric/gather/scatter modes.

# Returns
`GridBundle{L, typeof(grids[1])}` containing:
- `grids` — NTuple of output grids storing interpolated results.  
- `names` — Ordered list of all output quantity names, matching the grid tuple order.
"""
function PointSamples_interpolation(backend :: CPUComputeBackend, grid_template::PointSamples{3, TF}, input::InterpolationInput{3, TF}, catalog::InterpolationCatalog{3, N, G, Div, C, L}, itp_strategy::Type{ITPSTRATEGY} = itpSymmetric) where {N, G, Div, C, L, TF <: AbstractFloat, ITPSTRATEGY <: AbstractInterpolationStrategy}
    grids_result, LBVH, names, catalog_consice = initialize_interpolation(backend, grid_template, input, catalog)
    npoints = length(grid_template)
    @info "     SPH Interpolation: Start interpolation..."
    @inbounds @threads for i in 1:npoints
        # Do single point interpolation
        _point_samples_interpolation_kernel!(backend, grids_result, i, input, catalog_consice, LBVH, itp_strategy)

    end
    @info "     SPH Interpolation: End interpolation..."

    # Output (No extra operation, keep interface clean)
    grids = grids_result
    return GridBundle(grids, names)
end


# Interpolation kernels
## Need providing smoothed radius
@inline function _point_samples_interpolation_kernel!(:: CPUComputeBackend, grids :: NTuple{L, PointSamples{3, TF}}, i :: Int, input :: InterpolationInput{3, TF}, catalog_consice :: InterpolationCatalogConcise{3, N, G, Div, C}, LBVH :: LinearBVH, itp_strategy::Type{ITPSTRATEGY} = itpSymmetric) where {N, G, Div, C, L, TF <: AbstractFloat, ITPSTRATEGY <: AbstractInterpolationStrategy}
    # Get point
    @inbounds begin
        geometry = grids[1]
        
        xa = geometry.coor[1][i]
        ya = geometry.coor[2][i]
        za = geometry.coor[3][i]
        point :: NTuple{3, TF} = (xa, ya, za)
    end
    
    # Particles searching
    ha = LBVH_find_nearest_h(LBVH, point)

    # Interpolation
    itpresult :: Tuple{NTuple{N,TF}, NTuple{G,NTuple{3,TF}}, NTuple{Div,TF}, NTuple{C,NTuple{3,TF}}} = _general_quantity_interpolate_kernel(input, point, ha, LBVH, catalog_consice, itp_strategy)

    scalars :: NTuple{N,TF} = itpresult[1]
    gradients :: NTuple{G,NTuple{3,TF}} = itpresult[2]
    divergences :: NTuple{Div,TF} = itpresult[3]
    curls :: NTuple{C,NTuple{3,TF}} = itpresult[4]

    # Store results
    out_idx = 1

    ## Scalars
    if N > 0
        @inbounds for j in 1:N
            grids[out_idx].grid[i] = scalars[j]
            out_idx += 1
        end
    end

    ## Gradients
    if G > 0
        @inbounds for j in 1:G
            grad_quant = gradients[j]
            @inbounds for d in 1:3
                grids[out_idx].grid[i] = grad_quant[d]
                out_idx += 1
            end
        end
    end

    ## Divergences
    if Div > 0
        @inbounds for j in 1:Div
            div_quant = divergences[j]
            grids[out_idx].grid[i] = div_quant
            out_idx += 1
        end
    end

    ## Curls
    if C > 0
        @inbounds for j in 1:C
            curl_quant = curls[j]
            @inbounds for d in 1:3
                grids[out_idx].grid[i] = curl_quant[d]
                out_idx += 1
            end
        end
    end

    @assert out_idx == L + 1 "InterpolationCatalog ordering mismatch: expected $L values, stored $(out_idx - 1)"
    return nothing
end

## Not providing smoothed radius
@inline function _point_samples_interpolation_kernel!(:: CPUComputeBackend, grids :: NTuple{L, PointSamples{3, TF}}, i :: Int, input :: InterpolationInput{3, TF}, catalog_consice :: InterpolationCatalogConcise{3, N, G, Div, C}, LBVH :: LinearBVH, ::Type{itpScatter}) where {N, G, Div, C, L, TF <: AbstractFloat}
    # Get point
    @inbounds begin
        geometry = grids[1]

        xa = geometry.coor[1][i]
        ya = geometry.coor[2][i]
        za = geometry.coor[3][i]
        point :: NTuple{3, TF} = (xa, ya, za)
    end

    # Interpolation
    itpresult :: Tuple{NTuple{N,TF}, NTuple{G,NTuple{3,TF}}, NTuple{Div,TF}, NTuple{C,NTuple{3,TF}}} = _general_quantity_interpolate_kernel(input, point, LBVH, catalog_consice, itpScatter)

    scalars :: NTuple{N,TF} = itpresult[1]
    gradients :: NTuple{G,NTuple{3,TF}} = itpresult[2]
    divergences :: NTuple{Div,TF} = itpresult[3]
    curls :: NTuple{C,NTuple{3,TF}} = itpresult[4]

    # Store results
    out_idx = 1

    ## Scalars
    if N > 0
        @inbounds for j in 1:N
            grids[out_idx].grid[i] = scalars[j]
            out_idx += 1
        end
    end

    ## Gradients
    if G > 0
        @inbounds for j in 1:G
            grad_quant = gradients[j]
            @inbounds for d in 1:3
                grids[out_idx].grid[i] = grad_quant[d]
                out_idx += 1
            end
        end
    end

    ## Divergences
    if Div > 0
        @inbounds for j in 1:Div
            div_quant = divergences[j]
            grids[out_idx].grid[i] = div_quant
            out_idx += 1
        end
    end

    ## Curls
    if C > 0
        @inbounds for j in 1:C
            curl_quant = curls[j]
            @inbounds for d in 1:3
                grids[out_idx].grid[i] = curl_quant[d]
                out_idx += 1
            end
        end
    end

    @assert out_idx == L + 1 "InterpolationCatalog ordering mismatch: expected $L values, stored $(out_idx - 1)"
    return nothing
end
