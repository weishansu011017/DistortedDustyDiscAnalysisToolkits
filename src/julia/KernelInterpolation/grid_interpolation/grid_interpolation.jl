


function GeneralGridInterpolation(grid_template::GeneralGrid{D}, input::ITPINPUT, catalog::InterpolationCatalog{N, G, Div, C, L}, itp_strategy::InterpolationStrategy = itpSymmetric) where {D, N, G, Div, C, L, T <: AbstractFloat, ITPINPUT <: InterpolationInput{T}}
    # Generate a grid array for result
    order = catalog.ordered_names
    grids = ntuple(_ -> similar(grid_template), Val(L))
    gridv = grid_template.coor

    # Generate a Linear BVH structure for neighborhood searching
    LBVH = LinearBVH!(input, CodeType = UInt64)

    # Constructing empty pools and stacks for neighborhood searching
    npart = get_npart(input)
    pool_capacity = max(1, npart)
    pools = [zeros(Int, pool_capacity) for _ in 1:nthreads()]
    stack_capacity = max(1, 2 * npart + 6)                # 2 * (npart - 1) + 8
    stacks = [zeros(Int, stack_capacity) for _ in 1:nthreads()]

    # Get the multiplier of smoother radius for interpolation
    multiplier = KernelFunctionValid(input.smoothed_kernel, T)
    if itp_strategy == itpSymmetric || itp_strategy == itpScatter
        multiplier *= T(2.5)
    elseif itp_strategy == itpGather
        multiplier *= T(1.5)
    else
        throw(ArgumentError("Unknown interpolation strategy: $(itp_strategy)"))
    end
    h_values = input.h
    mean_h = isempty(h_values) ? zero(T) : sum(h_values) / length(h_values)
    gather_radius = multiplier * mean_h

    # Constructing an empty space for scalar interpolations
    scalar_count = length(catalog.scalar_slots)
    scalar_workspaces = [zeros(T, scalar_count) for _ in 1:nthreads()]

    @time begin
    @inbounds @threads for i in eachindex(gridv)
        # Get temp space
        tid = threadid()
        if tid > length(pools)
            tid = 1
        end
        pool = pools[tid]
        stack = stacks[tid]
        workspace = scalar_workspaces[tid]

        # Get the target
        point = gridv[i]

        # Particles searching
        selection = LBVH_query!(pool, stack, LBVH, point, gather_radius)
        ha = input.h[selection.nearest]
        selection = LBVH_query!(pool, stack, LBVH, point, multiplier * ha)

        # Interpolation
        out_idx = 1        
        ## Scalar interpolation
        if scalar_count > 0
            quantities_interpolate!(workspace, input, point, ha, selection, catalog.scalar_slots, catalog.scalar_snormalization, itp_strategy)
            @inbounds for j in eachindex(workspace)
                grids[out_idx].grid[i] = workspace[j]
                out_idx += 1
            end
        end

        ## Gradients interpolation
        for j in 1:G
            slot = catalog.grad_slots[j]
            grad_quant = slot == 0 ? gradient_density(input, point, ha, selection, itp_strategy) : gradient_quantity_interpolate(input, point, ha, selection, slot, itp_strategy)
            @inbounds for d in 1:D
                grids[out_idx].grid[i] = grad_quant[d]
                out_idx += 1
            end
        end

        ## Divergences interpolation
        for j in 1:Div
            slot = catalog.div_slots[j]
            ax, ay, az = slot
            div_quant = divergence_quantity_interpolate(input, point, ha, selection, ax, ay, az, itp_strategy)
            grids[out_idx].grid[i] = div_quant
            out_idx += 1
        end

        ## Curls interpolation
        for j in 1:C
            slot = catalog.curl_slots[j]
            ax, ay, az = slot
            curl_quant = curl_quantity_interpolate(input, point, ha, selection, ax, ay, az, itp_strategy)
            @inbounds for d in 1:D
                grids[out_idx].grid[i] = curl_quant[d]
                out_idx += 1
            end
        end

        @assert out_idx == L + 1 "InterpolationCatalog ordering mismatch: expected $L values, stored $(out_idx - 1)"
    end
    end

    return GridInterpolationResult{L}(grids, order)
end
