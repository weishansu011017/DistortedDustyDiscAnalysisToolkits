@inline function _LOS_density_kernel(input::ITPINPUT, reference_point::NTuple{2, T}, ha :: T, LBVH :: LinearBVH, :: Type{itpGather} = itpGather) where {ITPINPUT <: AbstractInterpolationInput, T <: AbstractFloat}
    # Prepare for interpolation
    K = input.smoothed_kernel
    Ktyp = typeof(K)
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

    radius = Kvalid * ha
    radius2 = radius * radius

    # Handle empty tree
    if iszero(root)
        nleaf = length(leaf_min[1])
        @inbounds for leaf_idx in 1:nleaf
            d2 = NeighborSearch._dist2_to_leaf_aabb(leaf_min, leaf_max, reference_point, leaf_idx)
            if d2 <= radius2
                ########### Found a neighbor, do accumulation ###########
                @inbounds begin
                    rb = (input.x[leaf_idx], input.y[leaf_idx])
                    mb = input.m[leaf_idx]
                    Sigma += _LOS_density_accumulation(reference_point, rb, mb, ha, K)
                end
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
                    @inbounds begin
                        rb = (input.x[leaf_idx], input.y[leaf_idx])
                        mb = input.m[leaf_idx]
                        Sigma += _LOS_density_accumulation(reference_point, rb, mb, ha, K)
                    end
                    #########################################################
                end
            end
            if RR[node]
                @inbounds leaf_idx = R[node]
                d2 = NeighborSearch._dist2_to_leaf_aabb(leaf_min, leaf_max, reference_point, leaf_idx)
                if d2 <= radius2
                    ########### Found a neighbor, do accumulation ###########
                    @inbounds begin
                        rb = (input.x[leaf_idx], input.y[leaf_idx])
                        mb = input.m[leaf_idx]
                        Sigma += _LOS_density_accumulation(reference_point, rb, mb, ha, K)
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
    return Sigma
end

@inline function _LOS_density_kernel(input::ITPINPUT, reference_point::NTuple{2, T}, ha :: T, LBVH :: LinearBVH, :: Type{itpScatter}) where {ITPINPUT <: AbstractInterpolationInput, T <: AbstractFloat}
    K = input.smoothed_kernel
    Ktyp = typeof(K)
    Kvalid = KernelFunctionValid(Ktyp, T)

    Sigma = zero(T)

    node_min = LBVH.node_aabb.min
    node_max = LBVH.node_aabb.max
    leaf_min = LBVH.leaf_aabb.min
    leaf_max = LBVH.leaf_aabb.max
    node_hmax = LBVH.node_hmax

    L  = LBVH.brt.left_child
    R  = LBVH.brt.right_child
    LL = LBVH.brt.is_leaf_left
    RR = LBVH.brt.is_leaf_right
    node_parent = LBVH.brt.node_parent
    root = LBVH.root

    if iszero(root)
        nleaf = length(leaf_min[1])
        @inbounds for leaf_idx in 1:nleaf
            hb = input.h[leaf_idx]
            radius = Kvalid * hb
            radius2 = radius * radius
            d2 = NeighborSearch._dist2_to_leaf_aabb(leaf_min, leaf_max, reference_point, leaf_idx)
            if d2 <= radius2
                @inbounds begin
                    rb = (input.x[leaf_idx], input.y[leaf_idx])
                    mb = input.m[leaf_idx]
                    Sigma += _LOS_density_accumulation(reference_point, rb, mb, hb, K)
                end
            end
        end
        return Sigma
    end

    node = root
    while node != 0
        radius_node = Kvalid * node_hmax[node]
        radius2_node = radius_node * radius_node
        dist2_node = NeighborSearch._dist2_to_node_aabb(node_min, node_max, reference_point, node)
        if dist2_node <= radius2_node
            if LL[node]
                @inbounds leaf_idx = L[node]
                hb = input.h[leaf_idx]
                radius = Kvalid * hb
                radius2 = radius * radius
                d2 = NeighborSearch._dist2_to_leaf_aabb(leaf_min, leaf_max, reference_point, leaf_idx)
                if d2 <= radius2
                    @inbounds begin
                        rb = (input.x[leaf_idx], input.y[leaf_idx])
                        mb = input.m[leaf_idx]
                        Sigma += _LOS_density_accumulation(reference_point, rb, mb, hb, K)
                    end
                end
            end
            if RR[node]
                @inbounds leaf_idx = R[node]
                hb = input.h[leaf_idx]
                radius = Kvalid * hb
                radius2 = radius * radius
                d2 = NeighborSearch._dist2_to_leaf_aabb(leaf_min, leaf_max, reference_point, leaf_idx)
                if d2 <= radius2
                    @inbounds begin
                        rb = (input.x[leaf_idx], input.y[leaf_idx])
                        mb = input.m[leaf_idx]
                        Sigma += _LOS_density_accumulation(reference_point, rb, mb, hb, K)
                    end
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

@inline function _LOS_density_kernel(input::ITPINPUT, reference_point::NTuple{2, T}, ha :: T, LBVH :: LinearBVH, :: Type{itpSymmetric}) where {ITPINPUT <: AbstractInterpolationInput, T <: AbstractFloat}
    K = input.smoothed_kernel
    Ktyp = typeof(K)
    Kvalid = KernelFunctionValid(Ktyp, T)

    Sigma = zero(T)

    node_min = LBVH.node_aabb.min
    node_max = LBVH.node_aabb.max
    leaf_min = LBVH.leaf_aabb.min
    leaf_max = LBVH.leaf_aabb.max
    node_hmax = LBVH.node_hmax

    L  = LBVH.brt.left_child
    R  = LBVH.brt.right_child
    LL = LBVH.brt.is_leaf_left
    RR = LBVH.brt.is_leaf_right
    node_parent = LBVH.brt.node_parent
    root = LBVH.root

    if iszero(root)
        nleaf = length(leaf_min[1])
        @inbounds for leaf_idx in 1:nleaf
            hb = input.h[leaf_idx]
            radius = Kvalid * max(ha, hb)
            radius2 = radius * radius
            d2 = NeighborSearch._dist2_to_leaf_aabb(leaf_min, leaf_max, reference_point, leaf_idx)
            if d2 <= radius2
                @inbounds begin
                    rb = (input.x[leaf_idx], input.y[leaf_idx])
                    mb = input.m[leaf_idx]
                    Sigma += _LOS_density_accumulation(reference_point, rb, mb, ha, hb, K)
                end
            end
        end
        return Sigma
    end

    node = root
    while node != 0
        radius_node = Kvalid * max(ha, node_hmax[node])
        radius2_node = radius_node * radius_node
        dist2_node = NeighborSearch._dist2_to_node_aabb(node_min, node_max, reference_point, node)
        if dist2_node <= radius2_node
            if LL[node]
                @inbounds leaf_idx = L[node]
                hb = input.h[leaf_idx]
                radius = Kvalid * max(ha, hb)
                radius2 = radius * radius
                d2 = NeighborSearch._dist2_to_leaf_aabb(leaf_min, leaf_max, reference_point, leaf_idx)
                if d2 <= radius2
                    @inbounds begin
                        rb = (input.x[leaf_idx], input.y[leaf_idx])
                        mb = input.m[leaf_idx]
                        Sigma += _LOS_density_accumulation(reference_point, rb, mb, ha, hb, K)
                    end
                end
            end
            if RR[node]
                @inbounds leaf_idx = R[node]
                hb = input.h[leaf_idx]
                radius = Kvalid * max(ha, hb)
                radius2 = radius * radius
                d2 = NeighborSearch._dist2_to_leaf_aabb(leaf_min, leaf_max, reference_point, leaf_idx)
                if d2 <= radius2
                    @inbounds begin
                        rb = (input.x[leaf_idx], input.y[leaf_idx])
                        mb = input.m[leaf_idx]
                        Sigma += _LOS_density_accumulation(reference_point, rb, mb, ha, hb, K)
                    end
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

## Multi-column LOS interpolation

@inline function _LOS_quantities_interpolate_kernel(input::ITPINPUT, reference_point::NTuple{2, T}, ha :: T, LBVH :: LinearBVH, columns::NTuple{M,Int}, ShepardNormalization :: NTuple{M, Bool}, :: Type{itpGather} = itpGather) where {ITPINPUT <: AbstractInterpolationInput, T <: AbstractFloat, M}
    # Prepare for interpolation
    K = input.smoothed_kernel
    Ktyp = typeof(K)
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

    radius = Kvalid * ha
    radius2 = radius * radius

    # Handle empty tree
    if iszero(root)
        nleaf = length(leaf_min[1])
        @inbounds for leaf_idx in 1:nleaf
            d2 = NeighborSearch._dist2_to_leaf_aabb(leaf_min, leaf_max, reference_point, leaf_idx)
            if d2 <= radius2
                ########### Found a neighbor, do accumulation ###########
                @inbounds begin
                    rb = (input.x[leaf_idx], input.y[leaf_idx])
                    mb = input.m[leaf_idx]
                    ρb = input.ρ[leaf_idx]
                    mWlρ += _LOS_ShepardNormalization_accumulation(reference_point, rb, mb, ρb, ha, K)
                    @inbounds for j in 1:M
                        column_idx = columns[j]
                        Ab = input.quant[column_idx][leaf_idx]
                        output[j] += _LOS_quantity_interpolate_accumulation(reference_point, rb, mb, ρb, Ab, ha, K)
                    end
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
                    @inbounds begin
                        rb = (input.x[leaf_idx], input.y[leaf_idx])
                        mb = input.m[leaf_idx]
                        ρb = input.ρ[leaf_idx]
                        mWlρ += _LOS_ShepardNormalization_accumulation(reference_point, rb, mb, ρb, ha, K)
                        @inbounds for j in 1:M
                            column_idx = columns[j]
                            Ab = input.quant[column_idx][leaf_idx]
                            output[j] += _LOS_quantity_interpolate_accumulation(reference_point, rb, mb, ρb, Ab, ha, K)
                        end
                    end
                    #########################################################
                end
            end
            if RR[node]
                @inbounds leaf_idx = R[node]
                d2 = NeighborSearch._dist2_to_leaf_aabb(leaf_min, leaf_max, reference_point, leaf_idx)
                if d2 <= radius2
                    ########### Found a neighbor, do accumulation ###########
                    @inbounds begin
                        rb = (input.x[leaf_idx], input.y[leaf_idx])
                        mb = input.m[leaf_idx]
                        ρb = input.ρ[leaf_idx]
                        mWlρ += _LOS_ShepardNormalization_accumulation(reference_point, rb, mb, ρb, ha, K)
                        @inbounds for j in 1:M
                            column_idx = columns[j]
                            Ab = input.quant[column_idx][leaf_idx]
                            output[j] += _LOS_quantity_interpolate_accumulation(reference_point, rb, mb, ρb, Ab, ha, K)
                        end
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

@inline function _LOS_quantities_interpolate_kernel(input::ITPINPUT, reference_point::NTuple{2, T}, ha :: T, LBVH :: LinearBVH, columns::NTuple{M,Int}, ShepardNormalization :: NTuple{M, Bool}, :: Type{itpScatter}) where {ITPINPUT <: AbstractInterpolationInput, T <: AbstractFloat, M}
    K = input.smoothed_kernel
    Ktyp = typeof(K)
    Kvalid = KernelFunctionValid(Ktyp, T)

    mWlρ :: T = zero(T)
    output :: MVector{M, T} = zero(MVector{M, T})

    node_min = LBVH.node_aabb.min
    node_max = LBVH.node_aabb.max
    leaf_min = LBVH.leaf_aabb.min
    leaf_max = LBVH.leaf_aabb.max
    node_hmax = LBVH.node_hmax

    L  = LBVH.brt.left_child
    R  = LBVH.brt.right_child
    LL = LBVH.brt.is_leaf_left
    RR = LBVH.brt.is_leaf_right
    node_parent = LBVH.brt.node_parent
    root = LBVH.root

    if iszero(root)
        nleaf = length(leaf_min[1])
        @inbounds for leaf_idx in 1:nleaf
            hb = input.h[leaf_idx]
            radius = Kvalid * hb
            radius2 = radius * radius
            d2 = NeighborSearch._dist2_to_leaf_aabb(leaf_min, leaf_max, reference_point, leaf_idx)
            if d2 <= radius2
                @inbounds begin
                    rb = (input.x[leaf_idx], input.y[leaf_idx])
                    mb = input.m[leaf_idx]
                    ρb = input.ρ[leaf_idx]
                    mWlρ += _LOS_ShepardNormalization_accumulation(reference_point, rb, mb, ρb, hb, K)
                    @inbounds for j in 1:M
                        column_idx = columns[j]
                        Ab = input.quant[column_idx][leaf_idx]
                        output[j] += _LOS_quantity_interpolate_accumulation(reference_point, rb, mb, ρb, Ab, hb, K)
                    end
                end
            end
        end
        @inbounds for j in 1:M
            if ShepardNormalization[j]
                output[j] /= mWlρ
            end
        end
        return output
    end

    node = root
    while node != 0
        radius_node = Kvalid * node_hmax[node]
        radius2_node = radius_node * radius_node
        dist2_node = NeighborSearch._dist2_to_node_aabb(node_min, node_max, reference_point, node)
        if dist2_node <= radius2_node
            if LL[node]
                @inbounds leaf_idx = L[node]
                hb = input.h[leaf_idx]
                radius = Kvalid * hb
                radius2 = radius * radius
                d2 = NeighborSearch._dist2_to_leaf_aabb(leaf_min, leaf_max, reference_point, leaf_idx)
                if d2 <= radius2
                    @inbounds begin
                        rb = (input.x[leaf_idx], input.y[leaf_idx])
                        mb = input.m[leaf_idx]
                        ρb = input.ρ[leaf_idx]
                        mWlρ += _LOS_ShepardNormalization_accumulation(reference_point, rb, mb, ρb, hb, K)
                        @inbounds for j in 1:M
                            column_idx = columns[j]
                            Ab = input.quant[column_idx][leaf_idx]
                            output[j] += _LOS_quantity_interpolate_accumulation(reference_point, rb, mb, ρb, Ab, hb, K)
                        end
                    end
                end
            end
            if RR[node]
                @inbounds leaf_idx = R[node]
                hb = input.h[leaf_idx]
                radius = Kvalid * hb
                radius2 = radius * radius
                d2 = NeighborSearch._dist2_to_leaf_aabb(leaf_min, leaf_max, reference_point, leaf_idx)
                if d2 <= radius2
                    @inbounds begin
                        rb = (input.x[leaf_idx], input.y[leaf_idx])
                        mb = input.m[leaf_idx]
                        ρb = input.ρ[leaf_idx]
                        mWlρ += _LOS_ShepardNormalization_accumulation(reference_point, rb, mb, ρb, hb, K)
                        @inbounds for j in 1:M
                            column_idx = columns[j]
                            Ab = input.quant[column_idx][leaf_idx]
                            output[j] += _LOS_quantity_interpolate_accumulation(reference_point, rb, mb, ρb, Ab, hb, K)
                        end
                    end
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

    @inbounds for j in 1:M
        if ShepardNormalization[j]
            output[j] /= mWlρ
        end
    end
    return output
end

@inline function _LOS_quantities_interpolate_kernel(input::ITPINPUT, reference_point::NTuple{2, T}, ha :: T, LBVH :: LinearBVH, columns::NTuple{M,Int}, ShepardNormalization :: NTuple{M, Bool}, :: Type{itpSymmetric}) where {ITPINPUT <: AbstractInterpolationInput, T <: AbstractFloat, M}
    K = input.smoothed_kernel
    Ktyp = typeof(K)
    Kvalid = KernelFunctionValid(Ktyp, T)

    mWlρ :: T = zero(T)
    output :: MVector{M, T} = zero(MVector{M, T})

    node_min = LBVH.node_aabb.min
    node_max = LBVH.node_aabb.max
    leaf_min = LBVH.leaf_aabb.min
    leaf_max = LBVH.leaf_aabb.max
    node_hmax = LBVH.node_hmax

    L  = LBVH.brt.left_child
    R  = LBVH.brt.right_child
    LL = LBVH.brt.is_leaf_left
    RR = LBVH.brt.is_leaf_right
    node_parent = LBVH.brt.node_parent
    root = LBVH.root

    if iszero(root)
        nleaf = length(leaf_min[1])
        @inbounds for leaf_idx in 1:nleaf
            hb = input.h[leaf_idx]
            hsym = max(ha, hb)
            radius = Kvalid * hsym
            radius2 = radius * radius
            d2 = NeighborSearch._dist2_to_leaf_aabb(leaf_min, leaf_max, reference_point, leaf_idx)
            if d2 <= radius2
                @inbounds begin
                    rb = (input.x[leaf_idx], input.y[leaf_idx])
                    mb = input.m[leaf_idx]
                    ρb = input.ρ[leaf_idx]
                    mWlρ += _LOS_ShepardNormalization_accumulation(reference_point, rb, mb, ρb, ha, hb, K)
                    @inbounds for j in 1:M
                        column_idx = columns[j]
                        Ab = input.quant[column_idx][leaf_idx]
                        output[j] += _LOS_quantity_interpolate_accumulation(reference_point, rb, mb, ρb, Ab, ha, hb, K)
                    end
                end
            end
        end
        @inbounds for j in 1:M
            if ShepardNormalization[j]
                output[j] /= mWlρ
            end
        end
        return output
    end

    node = root
    while node != 0
        radius_node = Kvalid * max(ha, node_hmax[node])
        radius2_node = radius_node * radius_node
        dist2_node = NeighborSearch._dist2_to_node_aabb(node_min, node_max, reference_point, node)
        if dist2_node <= radius2_node
            if LL[node]
                @inbounds leaf_idx = L[node]
                hb = input.h[leaf_idx]
                hsym = max(ha, hb)
                radius = Kvalid * hsym
                radius2 = radius * radius
                d2 = NeighborSearch._dist2_to_leaf_aabb(leaf_min, leaf_max, reference_point, leaf_idx)
                if d2 <= radius2
                    @inbounds begin
                        rb = (input.x[leaf_idx], input.y[leaf_idx])
                        mb = input.m[leaf_idx]
                        ρb = input.ρ[leaf_idx]
                        mWlρ += _LOS_ShepardNormalization_accumulation(reference_point, rb, mb, ρb, ha, hb, K)
                        @inbounds for j in 1:M
                            column_idx = columns[j]
                            Ab = input.quant[column_idx][leaf_idx]
                            output[j] += _LOS_quantity_interpolate_accumulation(reference_point, rb, mb, ρb, Ab, ha, hb, K)
                        end
                    end
                end
            end
            if RR[node]
                @inbounds leaf_idx = R[node]
                hb = input.h[leaf_idx]
                hsym = max(ha, hb)
                radius = Kvalid * hsym
                radius2 = radius * radius
                d2 = NeighborSearch._dist2_to_leaf_aabb(leaf_min, leaf_max, reference_point, leaf_idx)
                if d2 <= radius2
                    @inbounds begin
                        rb = (input.x[leaf_idx], input.y[leaf_idx])
                        mb = input.m[leaf_idx]
                        ρb = input.ρ[leaf_idx]
                        mWlρ += _LOS_ShepardNormalization_accumulation(reference_point, rb, mb, ρb, ha, hb, K)
                        @inbounds for j in 1:M
                            column_idx = columns[j]
                            Ab = input.quant[column_idx][leaf_idx]
                            output[j] += _LOS_quantity_interpolate_accumulation(reference_point, rb, mb, ρb, Ab, ha, hb, K)
                        end
                    end
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
