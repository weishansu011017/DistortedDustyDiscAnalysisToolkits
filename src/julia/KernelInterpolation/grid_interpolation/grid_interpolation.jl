
function GeneralGridInterpolation(grid::GeneralGrid{D}, input::ITPINPUT, catalog::InterpolationCatalog{N, G, Div, C, L}, itp_strategy::InterpolationStrategy = itpSymmetric) where {D, N, G, Div, C, L, T <: AbstractFloat, ITPINPUT <: InterpolationInput{T}}
    order = catalog.ordered_names
    grids = ntuple(_ -> similar(grid), Val(L))
    grid_coords = grid.coor

    LBVH = LinearBVH!(input, CodeType = UInt64)

    multiplier = KernelFunctionValid(input.smoothed_kernel, T)
    h_values = input.h
    mean_h = isempty(h_values) ? zero(T) : sum(h_values) / length(h_values)
    gather_radius = multiplier * mean_h

    scalar_count = length(catalog.scalar_slots)
    scalar_workspace = zeros(T, scalar_count)

    pool_capacity = max(1, length(input.x))
    pool = zeros(Int, pool_capacity)
    stack_capacity = max(1, 2 * length(LBVH.brt.left_child) + 8)
    stack = Vector{Int}(undef, stack_capacity)

    @inbounds for i in eachindex(grid_coords)
        point = grid_coords[i]

        selection = NeighborSelection(pool, zero(Int), zero(Int))
        ha = zero(T)

        if itp_strategy == itpSymmetric || itp_strategy == itpScatter
            selection, ha = LBVH_query!(pool, stack, LBVH, point, multiplier, h_values)
        elseif itp_strategy == itpGather
            selection = LBVH_query!(pool, stack, LBVH, point, gather_radius)
            if selection.count != 0
                nearest = nearest_index(selection)
                ha = h_values[nearest]
                radius = multiplier * ha
                selection = LBVH_query!(pool, stack, LBVH, point, radius)
                nearest = selection.count == 0 ? nearest : nearest_index(selection)
                ha = h_values[nearest]
            end
        else
            throw(ArgumentError("Unknown interpolation strategy: $(itp_strategy)"))
        end

        out_idx = 1
        if scalar_count > 0
            quantities_interpolate!(scalar_workspace, input, point, ha, selection, catalog.scalar_slots, itp_strategy)
            @inbounds for j in 1:scalar_count
                grids[out_idx].grid[i] = scalar_workspace[j]
                out_idx += 1
            end
        end

        for j in 1:G
            slot = catalog.grad_slots[j]
            grad_quant = slot == 0 ? gradient_density(input, point, ha, selection, itp_strategy) : gradient_quantity_interpolate(input, point, ha, selection, slot, itp_strategy)
            @inbounds for d in 1:D
                grids[out_idx].grid[i] = grad_quant[d]
                out_idx += 1
            end
        end

        for j in 1:Div
            slot = catalog.div_slots[j]
            ax, ay, az = slot
            div_quant = divergence_quantity_interpolate(input, point, ha, selection, ax, ay, az, itp_strategy)
            grids[out_idx].grid[i] = div_quant
            out_idx += 1
        end

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

    return grids, order
end
