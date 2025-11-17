

@inline function _general_grid_interpolation_kernel!(grids :: NTuple{L, GeneralGrid{D}}, input :: ITPINPUT, catalog_consice :: InterpolationCatalogConcise{N, G, Div, C}, LBVH :: LinearBVH, itp_strategy::Type{ITPSTRATEGY} = itpSymmetric) where {D, N, G, Div, C, L, TF <: AbstractFloat, VF <: CuDeviceVector{TF}, ITPINPUT <: InterpolationInput{TF, VF}, ITPSTRATEGY <: AbstractInterpolationStrategy}
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
        point = grids[1].coor[i]
        # Particles searching
        ha_nearest_idx, _ = LBVH_find_nearest(LBVH, point)
        ha = input.h[ha_nearest_idx]

        # Interpolation
        out_idx = 1        
        ## Scalar interpolation
        if N > 0
            scalars = PhantomRevealer.quantities_interpolate(input, point, ha, LBVH, catalog_consice.scalar_slots, catalog_consice.scalar_snormalization, itp_strategy)
            @inbounds for j in 1:N
                grids[out_idx].grid[i] = scalars[j]
                out_idx += 1
            end
        end

        ## Gradients interpolation
        for j in 1:G
            slot = catalog_consice.grad_slots[j]
            grad_quant = slot == 0 ? PhantomRevealer.gradient_density(input, point, ha, LBVH, itp_strategy) : PhantomRevealer.gradient_quantity_interpolate(input, point, ha, LBVH, slot, itp_strategy)
            @inbounds for d in 1:D
                grids[out_idx].grid[i] = grad_quant[d]
                out_idx += 1
            end
        end

        ## Divergences interpolation
        for j in 1:Div
            slot = catalog_consice.div_slots[j]
            ax, ay, az = slot
            div_quant = PhantomRevealer.divergence_quantity_interpolate(input, point, ha, LBVH, ax, ay, az, itp_strategy)
            grids[out_idx].grid[i] = div_quant
            out_idx += 1
        end

        ## Curls interpolation
        for j in 1:C
            slot = catalog_consice.curl_slots[j]
            ax, ay, az = slot
            curl_quant = PhantomRevealer.curl_quantity_interpolate(input, point, ha, LBVH, ax, ay, az, itp_strategy)
            @inbounds for d in 1:D
                grids[out_idx].grid[i] = curl_quant[d]
                out_idx += 1
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
    grids, LBVH, order, catalog_consice = PhantomRevealer.initialize_interpolation(PhantomRevealer.CPUComputeBackend(), grid_template, input, catalog)
    # To CuVector
    input_Cu = to_CuVector(input)
    grids_Cu = ntuple(i -> to_CuVector(grids[i]), Val(L))
    LBVH_Cu = to_CuVector(LBVH)

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
    return GridInterpolationResult{L}(grids_result, order)
end