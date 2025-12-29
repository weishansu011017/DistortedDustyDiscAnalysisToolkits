@inline function _general_grid_interpolation_kernel!(grids :: NTuple{L, GeneralGrid{3}}, input :: ITPINPUT, catalog_consice :: InterpolationCatalogConcise{N, G, Div, C}, LBVH :: LinearBVH, itp_strategy::Type{ITPSTRATEGY} = itpSymmetric) where {N, G, Div, C, L, TF <: AbstractFloat, ITPINPUT <: InterpolationInput{TF}, ITPSTRATEGY <: AbstractInterpolationStrategy}
    tid    = Int(CUDA.threadIdx().x)
    bid    = Int(CUDA.blockIdx().x)
    bdim   = Int(CUDA.blockDim().x)
    gdim   = Int(CUDA.gridDim().x)

    gid    = (bid - 1) * bdim + tid
    stride = bdim * gdim

    npoints = length(grids[1])
    i = gid
    while i <= npoints
        # Get point
        @inbounds begin
            xa = grids[1].coor[1][i]
            ya = grids[1].coor[2][i]
            za = grids[1].coor[3][i]
            point :: NTuple{3, TF} = (xa, ya, za)
        end
        
        # Particles searching
        ha = LBVH_find_nearest_h(LBVH, point)

        # Interpolation
        itpresult :: Tuple{NTuple{N,TF}, NTuple{G,NTuple{3,TF}}, NTuple{Div,TF}, NTuple{C,NTuple{3,TF}}} = PhantomRevealer.KernelInterpolation._general_quantity_interpolate_kernel(input, point, ha, LBVH, catalog_consice, itp_strategy)

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
        i += stride
    end
    return nothing
end

@inline function _general_grid_interpolation_kernel!(grids :: NTuple{L, GeneralGrid{3}}, input :: ITPINPUT, catalog_consice :: InterpolationCatalogConcise{N, G, Div, C}, LBVH :: LinearBVH, ::Type{itpScatter}) where {N, G, Div, C, L, TF <: AbstractFloat, ITPINPUT <: InterpolationInput{TF}}
    tid    = Int(CUDA.threadIdx().x)
    bid    = Int(CUDA.blockIdx().x)
    bdim   = Int(CUDA.blockDim().x)
    gdim   = Int(CUDA.gridDim().x)

    gid    = (bid - 1) * bdim + tid
    stride = bdim * gdim

    npoints = length(grids[1])
    i = gid
    while i <= npoints
        # Get point
        @inbounds begin
            xa = grids[1].coor[1][i]
            ya = grids[1].coor[2][i]
            za = grids[1].coor[3][i]
            point :: NTuple{3, TF} = (xa, ya, za)
        end

        # Interpolation
        itpresult :: Tuple{NTuple{N,TF}, NTuple{G,NTuple{3,TF}}, NTuple{Div,TF}, NTuple{C,NTuple{3,TF}}} = PhantomRevealer.KernelInterpolation._general_quantity_interpolate_kernel(input, point, LBVH, catalog_consice, itpScatter)

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
        i += stride
    end
    return nothing
end

"""
    GeneralGrid_interpolation(backend::MetalComputeBackend, grid_template::GeneralGrid{D},
                          input::ITPINPUT, catalog::InterpolationCatalog{N, G, Div, C, L},
                          itp_strategy::Type{ITPSTRATEGY} = itpSymmetric)

Performs SPH grid interpolation on the GPU using Apple's Metal backend.
The routine copies particle data, grids, and BVH structures to Metal buffers,
launches the interpolation kernel, and copies results back to host memory.

# Parameters
- `backend::MetalComputeBackend`  
  GPU execution backend using Metal.

- `grid_template::GeneralGrid{D}`  
  Template grid defining dimensionality and coordinate storage for the output grids.

- `input::ITPINPUT`  
  `InterpolationInput` holding positions, smoothing lengths, quantities,
  and the SPH kernel to be used for GPU-side interpolation.

- `catalog::InterpolationCatalog{N, G, Div, C, L}`  
  Full interpolation catalog describing which scalar, gradient, divergence,
  and curl quantities should be computed.

- `itp_strategy::Type{ITPSTRATEGY}`  
  Interpolation strategy type (e.g., symmetric, gather, scatter).

# Returns
`GridInterpolationResult{L}` containing:
- `grids` — NTuple of host-side grids with all interpolated fields.  
- `order` — Ordered list of output quantity names.

"""
function PhantomRevealer.GeneralGrid_interpolation(:: CUDAComputeBackend, grid_template::GeneralGrid{D}, input::ITPINPUT, catalog::InterpolationCatalog{N, G, Div, C, L}, itp_strategy::Type{ITPSTRATEGY} = itpSymmetric) where {D, N, G, Div, C, L, T <: AbstractFloat, ITPINPUT <: InterpolationInput{T}, ITPSTRATEGY <: AbstractInterpolationStrategy}
    grids, LBVH, order, catalog_consice, p = PhantomRevealer.initialize_interpolation(PhantomRevealer.CPUComputeBackend(), grid_template, input, catalog)
    # To CuVector
    @info "Copying interpolated grids to device memory..."
    @time begin
    input_Cu = to_CuVector(input)
    grids_Cu = ntuple(i -> to_CuVector(grids[i]), Val(L))
    LBVH_Cu = to_CuVector(LBVH)
    end
    @info "End copying interpolated grids to device memory."

    npoints = length(grid_template)
    @info"Start interpolation..."
    @time begin
    @cuda threads=(256,) blocks=(cld(npoints, 256)) _general_grid_interpolation_kernel!(grids_Cu, input_Cu, catalog_consice, LBVH_Cu, itp_strategy)
    CUDA.synchronize()
    end
    @info"End interpolation."
    @info "Copying interpolated grids back to host memory..."
    @time begin
    grids_result = ntuple(i -> PhantomRevealer.to_HostVector(grids_Cu[i]), Val(L))
    end
    @info "End copying interpolated grids back to host memory."

    # Reorder grids back to original order
    @info "Reordering output grids back to original order..."
    @time begin
    # Reorder coor
    @inbounds for i in 1:D
        Base.invpermute!(grids_result[1].coor[i], p)
    end

    # Reorder each grid
    for grid in grids_result
        Base.invpermute!(grid.grid, p)
    end
    end
    @info "End reordering output grids..."

    return GridInterpolationResult{L}(grids_result, order)
end