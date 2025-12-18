@inline function _LOS_density_kernel(input::ITPINPUT, reference_point::NTuple{2, T}, ha :: T, LBVH :: LinearBVH, itp_strategy :: Type{itpGather} = itpGather) where {ITPINPUT <: AbstractInterpolationInput, T <: AbstractFloat}
    # Prepare for interpolation
    Ktyp = typeof(input.smoothed_kernel)
    Kvalid = KernelFunctionValid(Ktyp, T)

    # Initialize counter
    Sigma :: T = zero(T)

    # LBVH data
    node_min = LBVH.node_aabb.min
    node_max = LBVH.node_aabb.max
    leaf_min = LBVH.leaf_aabb.min
    leaf_max = LBVH.leaf_aabb.max

    L  = LBVH.brt.left_child
    R  = LBVH.brt.right_child
    LL = LBVH.brt.is_leaf_left
    RR = LBVH.brt.is_leaf_right
    node_parent = LBVH.brt.node_parent
    root = LBVH.root

    # Do traversal
    radius = Kvalid * ha
    radius2 = radius * radius

    # Handle empty tree
    if iszero(root)
        nleaf = length(leaf_min[1])
        @inbounds for leaf_idx in 1:nleaf
            d2 = NeighborSearch._dist2_to_leaf_aabb(leaf_min, leaf_max, reference_point, leaf_idx)
            if d2 <= radius2
                ########### Found a neighbor, do accumulation ###########
                Sigma += _LOS_density_accumulation(input, reference_point, ha, itp_strategy, leaf_idx)
                #########################################################
            end
        end
        return Sigma
    end

    # Start traversal
    node = root
    while node != 0
        dist2_node = NeighborSearch._dist2_to_node_aabb(node_min, node_max, reference_point, node)
        if dist2_node <= radius2
            if LL[node]
                @inbounds leaf_idx = L[node]
                d2 = NeighborSearch._dist2_to_leaf_aabb(leaf_min, leaf_max, reference_point, leaf_idx)
                if d2 <= radius2
                    ########### Found a neighbor, do accumulation ###########
                    Sigma += _LOS_density_accumulation(input, reference_point, ha, itp_strategy, leaf_idx)
                    #########################################################
                end
            end
            if RR[node]
                @inbounds leaf_idx = R[node]
                d2 = NeighborSearch._dist2_to_leaf_aabb(leaf_min, leaf_max, reference_point, leaf_idx)
                if d2 <= radius2
                    ########### Found a neighbor, do accumulation ###########
                    Sigma += _LOS_density_accumulation(input, reference_point, ha, itp_strategy, leaf_idx)
                    #########################################################
                end
            end

            if !LL[node]
                node = L[node]
                continue
            end
            if !RR[node]
                node = R[node]
                continue
            end

            node = NeighborSearch._next_internal_node(node, L, R, LL, RR, node_parent)
        else
            node = NeighborSearch._next_internal_node(node, L, R, LL, RR, node_parent)
        end
    end
    return Sigma
end

@inline function _LOS_quantities_interpolate_kernel!(output :: O, input::ITPINPUT, reference_point::NTuple{2, T}, ha :: T, LBVH :: LinearBVH, columns::NTuple{M,Int}, ShepardNormalization :: NTuple{M, Bool}, itp_strategy :: Type{itpGather} = itpGather) where {ITPINPUT <: AbstractInterpolationInput, T <: AbstractFloat, O<:AbstractVector{T}, M}
    @assert length(output) == M "Length of `output` must match the requested columns."
    # Prepare for interpolation
    Ktyp = typeof(input.smoothed_kernel)
    Kvalid = KernelFunctionValid(Ktyp, T)

    # Initialize counter
    mWlρ :: T = zero(T)
    fill!(output, zero(T))

    # LBVH data
    node_min = LBVH.node_aabb.min
    node_max = LBVH.node_aabb.max
    leaf_min = LBVH.leaf_aabb.min
    leaf_max = LBVH.leaf_aabb.max

    L  = LBVH.brt.left_child
    R  = LBVH.brt.right_child
    LL = LBVH.brt.is_leaf_left
    RR = LBVH.brt.is_leaf_right
    node_parent = LBVH.brt.node_parent
    root = LBVH.root

    # Do traversal
    radius = Kvalid * ha
    radius2 = radius * radius

    # Handle empty tree
    if iszero(root)
        nleaf = length(leaf_min[1])
        @inbounds for leaf_idx in 1:nleaf
            d2 = NeighborSearch._dist2_to_leaf_aabb(leaf_min, leaf_max, reference_point, leaf_idx)
            if d2 <= radius2
                ########### Found a neighbor, do accumulation ###########
                mWlρ += _LOS_ShepardNormalization_accumulation(input, reference_point, ha, itp_strategy, leaf_idx)
                @inbounds for j in 1:M
                    column_idx = columns[j]
                    output[j] += _LOS_quantity_interpolate_accumulation(input, reference_point, ha, column_idx, itp_strategy, leaf_idx)
                end
                #########################################################
            end
        end
        # Shepard normalization
        @inbounds for j in eachindex(output)
            if ShepardNormalization[j]
                output[j] /= mWlρ
            end
        end
    end

    # Start traversal
    node = root
    while node != 0
        dist2_node = NeighborSearch._dist2_to_node_aabb(node_min, node_max, reference_point, node)
        if dist2_node <= radius2
            if LL[node]
                @inbounds leaf_idx = L[node]
                d2 = NeighborSearch._dist2_to_leaf_aabb(leaf_min, leaf_max, reference_point, leaf_idx)
                if d2 <= radius2
                    ########### Found a neighbor, do accumulation ###########
                    mWlρ += _LOS_ShepardNormalization_accumulation(input, reference_point, ha, itp_strategy, leaf_idx)
                    @inbounds for j in 1:M
                        column_idx = columns[j]
                        output[j] += _LOS_quantity_interpolate_accumulation(input, reference_point, ha, column_idx, itp_strategy, leaf_idx)
                    end
                    #########################################################
                end
            end
            if RR[node]
                @inbounds leaf_idx = R[node]
                d2 = NeighborSearch._dist2_to_leaf_aabb(leaf_min, leaf_max, reference_point, leaf_idx)
                if d2 <= radius2
                    ########### Found a neighbor, do accumulation ###########
                    mWlρ += _LOS_ShepardNormalization_accumulation(input, reference_point, ha, itp_strategy, leaf_idx)
                    @inbounds for j in 1:M
                        column_idx = columns[j]
                        output[j] += _LOS_quantity_interpolate_accumulation(input, reference_point, ha, column_idx, itp_strategy, leaf_idx)
                    end
                    #########################################################
                end
            end

            if !LL[node]
                node = L[node]
                continue
            end
            if !RR[node]
                node = R[node]
                continue
            end

            node = NeighborSearch._next_internal_node(node, L, R, LL, RR, node_parent)
        else
            node = NeighborSearch._next_internal_node(node, L, R, LL, RR, node_parent)
        end
    end
    # Shepard normalization
    @inbounds for j in eachindex(output)
        if ShepardNormalization[j]
            output[j] /= mWlρ
        end
    end
    return nothing
end

@inline function _LOS_quantities_interpolate_kernel!(output :: O, input::ITPINPUT, reference_point::NTuple{2, T}, ha :: T, LBVH :: LinearBVH, itp_strategy :: Type{ITPSTRATEGY} = itpSymmetric) where {ITPINPUT <: AbstractInterpolationInput, T <: AbstractFloat, O<:AbstractVector{T}, ITPSTRATEGY <: AbstractInterpolationStrategy}
    val_len = Val(length(input.quant))
    columns = ntuple(identity, val_len)
    ShepardNormalization = ntuple(_ -> true, val_len)
    return _LOS_quantities_interpolate_kernel!(output, input, reference_point, ha, LBVH, columns, ShepardNormalization, itp_strategy)
end

@inline function _LOS_quantities_interpolate_kernel(input::ITPINPUT, reference_point::NTuple{2, T}, ha :: T, LBVH :: LinearBVH, columns::NTuple{M,Int}, ShepardNormalization :: NTuple{M, Bool}, itp_strategy :: Type{itpGather} = itpGather) where {ITPINPUT <: AbstractInterpolationInput, T <: AbstractFloat, M}
    # Prepare for interpolation
    Ktyp = typeof(input.smoothed_kernel)
    Kvalid = KernelFunctionValid(Ktyp, T)

    # Initialize counter
    mWlρ :: T = zero(T)
    output :: MVector{M, T} = zero(MVector{M, T})

    # LBVH data
    node_min = LBVH.node_aabb.min
    node_max = LBVH.node_aabb.max
    leaf_min = LBVH.leaf_aabb.min
    leaf_max = LBVH.leaf_aabb.max

    L  = LBVH.brt.left_child
    R  = LBVH.brt.right_child
    LL = LBVH.brt.is_leaf_left
    RR = LBVH.brt.is_leaf_right
    node_parent = LBVH.brt.node_parent
    root = LBVH.root

    # Do traversal
    radius = Kvalid * ha
    radius2 = radius * radius

    # Handle empty tree
    if iszero(root)
        nleaf = length(leaf_min[1])
        @inbounds for leaf_idx in 1:nleaf
            d2 = NeighborSearch._dist2_to_leaf_aabb(leaf_min, leaf_max, reference_point, leaf_idx)
            if d2 <= radius2
                ########### Found a neighbor, do accumulation ###########
                mWlρ += _LOS_ShepardNormalization_accumulation(input, reference_point, ha, itp_strategy, leaf_idx)
                @inbounds for j in 1:M
                    column_idx = columns[j]
                    output[j] += _LOS_quantity_interpolate_accumulation(input, reference_point, ha, column_idx, itp_strategy, leaf_idx)
                end
                #########################################################
            end
        end
        # Shepard normalization
        @inbounds for j in 1:M
            if ShepardNormalization[j]
                output[j] /= mWlρ
            end
        end
        return output
    end

    # Start traversal
    node = root
    while node != 0
        dist2_node = NeighborSearch._dist2_to_node_aabb(node_min, node_max, reference_point, node)
        if dist2_node <= radius2
            if LL[node]
                @inbounds leaf_idx = L[node]
                d2 = NeighborSearch._dist2_to_leaf_aabb(leaf_min, leaf_max, reference_point, leaf_idx)
                if d2 <= radius2
                    ########### Found a neighbor, do accumulation ###########
                    mWlρ += _LOS_ShepardNormalization_accumulation(input, reference_point, ha, itp_strategy, leaf_idx)
                    @inbounds for j in 1:M
                        column_idx = columns[j]
                        output[j] += _LOS_quantity_interpolate_accumulation(input, reference_point, ha, column_idx, itp_strategy, leaf_idx)
                    end
                    #########################################################
                end
            end
            if RR[node]
                @inbounds leaf_idx = R[node]
                d2 = NeighborSearch._dist2_to_leaf_aabb(leaf_min, leaf_max, reference_point, leaf_idx)
                if d2 <= radius2
                    ########### Found a neighbor, do accumulation ###########
                    mWlρ += _LOS_ShepardNormalization_accumulation(input, reference_point, ha, itp_strategy, leaf_idx)
                    @inbounds for j in 1:M
                        column_idx = columns[j]
                        output[j] += _LOS_quantity_interpolate_accumulation(input, reference_point, ha, column_idx, itp_strategy, leaf_idx)
                    end
                    #########################################################
                end
            end

            if !LL[node]
                node = L[node]
                continue
            end
            if !RR[node]
                node = R[node]
                continue
            end

            node = NeighborSearch._next_internal_node(node, L, R, LL, RR, node_parent)
        else
            node = NeighborSearch._next_internal_node(node, L, R, LL, RR, node_parent)
        end
    end
    # Shepard normalization
    @inbounds for j in 1:M
        if ShepardNormalization[j]
            output[j] /= mWlρ
        end
    end
    return output
end

@inline function _LOS_quantities_interpolate_kernel(input::ITPINPUT, reference_point::NTuple{2, T}, ha :: T, LBVH :: LinearBVH, itp_strategy :: Type{ITPSTRATEGY} = itpSymmetric) where {ITPINPUT <: AbstractInterpolationInput, T <: AbstractFloat, ITPSTRATEGY <: AbstractInterpolationStrategy}
    val_len = Val(length(input.quant))
    columns = ntuple(identity, val_len)
    ShepardNormalization = ntuple(_ -> true, val_len)
    return _LOS_quantities_interpolate_kernel(input, reference_point, ha, LBVH, columns, ShepardNormalization, itp_strategy)
end
