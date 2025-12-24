@inline function _divergence_quantity_interpolate_kernel(input::ITPINPUT, reference_point::NTuple{3, T}, ha :: T, LBVH :: LinearBVH, Ax_column_idx :: Int64, Ay_column_idx :: Int64, Az_column_idx :: Int64, itp_strategy :: Type{itpGather} = itpGather) where {ITPINPUT <: AbstractInterpolationInput, T <: AbstractFloat}
    # Prepare for interpolation
    Ktyp = typeof(input.smoothed_kernel)
    Kvalid = KernelFunctionValid(Ktyp, T)

    # Initialize counter
    ∇Af :: T = zero(T)
    ∇Axb :: T = zero(T)
    ∇Ayb :: T = zero(T)
    ∇Azb :: T = zero(T)

    mWlρ :: T = zero(T)
    Ax :: T = zero(T)
    Ay :: T = zero(T)
    Az :: T = zero(T)
    ρ :: T = zero(T)

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
                ∇AfW, ∇AxbW, ∇AybW, ∇AzbW = _divergence_quantity_accumulation(input, reference_point, ha, Ax_column_idx, Ay_column_idx, Az_column_idx, itp_strategy, leaf_idx)
                ρ += _density_accumulation(input, reference_point, ha, itp_strategy, leaf_idx)
                Ax += _quantity_interpolate_accumulation(input, reference_point, ha, Ax_column_idx, itp_strategy, leaf_idx)
                Ay += _quantity_interpolate_accumulation(input, reference_point, ha, Ay_column_idx, itp_strategy, leaf_idx)
                Az += _quantity_interpolate_accumulation(input, reference_point, ha, Az_column_idx, itp_strategy, leaf_idx)

                ∇Af += ∇AfW
                ∇Axb += ∇AxbW
                ∇Ayb += ∇AybW
                ∇Azb += ∇AzbW
                mWlρ += _ShepardNormalization_accumulation(input, reference_point, ha, itp_strategy, leaf_idx)
                #########################################################
            end
        end
        if iszero(ρ)
            return T(NaN)
        end

        # Shepard normalization
        Ax /= mWlρ
        Ay /= mWlρ
        Az /= mWlρ

        # Construct gradient
        ∇Ab = Ax * ∇Axb + Ay * ∇Ayb + Az * ∇Azb

        # Final result
        ∇A = (∇Af - ∇Ab)/ρ

        return ∇A
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
                    ∇AfW, ∇AxbW, ∇AybW, ∇AzbW = _divergence_quantity_accumulation(input, reference_point, ha, Ax_column_idx, Ay_column_idx, Az_column_idx, itp_strategy, leaf_idx)
                    ρ += _density_accumulation(input, reference_point, ha, itp_strategy, leaf_idx)
                    Ax += _quantity_interpolate_accumulation(input, reference_point, ha, Ax_column_idx, itp_strategy, leaf_idx)
                    Ay += _quantity_interpolate_accumulation(input, reference_point, ha, Ay_column_idx, itp_strategy, leaf_idx)
                    Az += _quantity_interpolate_accumulation(input, reference_point, ha, Az_column_idx, itp_strategy, leaf_idx)

                    ∇Af += ∇AfW
                    ∇Axb += ∇AxbW
                    ∇Ayb += ∇AybW
                    ∇Azb += ∇AzbW
                    mWlρ += _ShepardNormalization_accumulation(input, reference_point, ha, itp_strategy, leaf_idx)
                    #########################################################
                end
            end
            if RR[node]
                @inbounds leaf_idx = R[node]
                d2 = NeighborSearch._dist2_to_leaf_aabb(leaf_min, leaf_max, reference_point, leaf_idx)
                if d2 <= radius2
                    ########### Found a neighbor, do accumulation ###########
                    ∇AfW, ∇AxbW, ∇AybW, ∇AzbW = _divergence_quantity_accumulation(input, reference_point, ha, Ax_column_idx, Ay_column_idx, Az_column_idx, itp_strategy, leaf_idx)
                    ρ += _density_accumulation(input, reference_point, ha, itp_strategy, leaf_idx)
                    Ax += _quantity_interpolate_accumulation(input, reference_point, ha, Ax_column_idx, itp_strategy, leaf_idx)
                    Ay += _quantity_interpolate_accumulation(input, reference_point, ha, Ay_column_idx, itp_strategy, leaf_idx)
                    Az += _quantity_interpolate_accumulation(input, reference_point, ha, Az_column_idx, itp_strategy, leaf_idx)

                    ∇Af += ∇AfW
                    ∇Axb += ∇AxbW
                    ∇Ayb += ∇AybW
                    ∇Azb += ∇AzbW
                    mWlρ += _ShepardNormalization_accumulation(input, reference_point, ha, itp_strategy, leaf_idx)
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
    if iszero(ρ)
        return T(NaN)
    end

    # Shepard normalization
    Ax /= mWlρ
    Ay /= mWlρ
    Az /= mWlρ

    # Construct gradient
    ∇Ab = Ax * ∇Axb + Ay * ∇Ayb + Az * ∇Azb

    # Final result
    ∇A = (∇Af - ∇Ab)/ρ

    return ∇A
end

@inline function _divergence_quantity_interpolate_kernel(input::ITPINPUT, reference_point::NTuple{3, T}, ha :: T, LBVH :: LinearBVH, Ax_column_idx :: Int64, Ay_column_idx :: Int64, Az_column_idx :: Int64, itp_strategy :: Type{itpScatter}) where {ITPINPUT <: AbstractInterpolationInput, T <: AbstractFloat}
    Ktyp = typeof(input.smoothed_kernel)
    Kvalid = KernelFunctionValid(Ktyp, T)

    ∇Af = zero(T); ∇Axb = zero(T); ∇Ayb = zero(T); ∇Azb = zero(T)
    mWlρ = zero(T); Ax = zero(T); Ay = zero(T); Az = zero(T); ρ = zero(T)

    node_min = LBVH.node_aabb.min; node_max = LBVH.node_aabb.max
    leaf_min = LBVH.leaf_aabb.min; leaf_max = LBVH.leaf_aabb.max
    node_hmax = LBVH.node_hmax
    L = LBVH.brt.left_child; R = LBVH.brt.right_child
    LL = LBVH.brt.is_leaf_left; RR = LBVH.brt.is_leaf_right
    parent = LBVH.brt.node_parent; root = LBVH.root

    if iszero(root)
        nleaf = length(leaf_min[1])
        @inbounds for leaf_idx in 1:nleaf
            hb = input.h[leaf_idx]
            r = Kvalid * hb; r2 = r * r
            d2 = NeighborSearch._dist2_to_leaf_aabb(leaf_min, leaf_max, reference_point, leaf_idx)
            if d2 <= r2
                ∇AfW, ∇AxbW, ∇AybW, ∇AzbW = _divergence_quantity_accumulation(input, reference_point, hb, Ax_column_idx, Ay_column_idx, Az_column_idx, itp_strategy, leaf_idx)
                ρ += _density_accumulation(input, reference_point, hb, itp_strategy, leaf_idx)
                Ax += _quantity_interpolate_accumulation(input, reference_point, hb, Ax_column_idx, itp_strategy, leaf_idx)
                Ay += _quantity_interpolate_accumulation(input, reference_point, hb, Ay_column_idx, itp_strategy, leaf_idx)
                Az += _quantity_interpolate_accumulation(input, reference_point, hb, Az_column_idx, itp_strategy, leaf_idx)
                ∇Af += ∇AfW; ∇Axb += ∇AxbW; ∇Ayb += ∇AybW; ∇Azb += ∇AzbW
                mWlρ += _ShepardNormalization_accumulation(input, reference_point, hb, itp_strategy, leaf_idx)
            end
        end
        if iszero(ρ)
            return T(NaN)
        end
        Ax /= mWlρ; Ay /= mWlρ; Az /= mWlρ
        ∇Ab = Ax * ∇Axb + Ay * ∇Ayb + Az * ∇Azb
        return (∇Af - ∇Ab)/ρ
    end

    node = root
    while node != 0
        rnode = Kvalid * node_hmax[node]; r2node = rnode * rnode
        d2node = NeighborSearch._dist2_to_node_aabb(node_min, node_max, reference_point, node)
        if d2node <= r2node
            if LL[node]
                @inbounds leaf_idx = L[node]
                hb = input.h[leaf_idx]
                rleaf = Kvalid * hb; r2leaf = rleaf * rleaf
                d2 = NeighborSearch._dist2_to_leaf_aabb(leaf_min, leaf_max, reference_point, leaf_idx)
                if d2 <= r2leaf
                    ∇AfW, ∇AxbW, ∇AybW, ∇AzbW = _divergence_quantity_accumulation(input, reference_point, hb, Ax_column_idx, Ay_column_idx, Az_column_idx, itp_strategy, leaf_idx)
                    ρ += _density_accumulation(input, reference_point, hb, itp_strategy, leaf_idx)
                    Ax += _quantity_interpolate_accumulation(input, reference_point, hb, Ax_column_idx, itp_strategy, leaf_idx)
                    Ay += _quantity_interpolate_accumulation(input, reference_point, hb, Ay_column_idx, itp_strategy, leaf_idx)
                    Az += _quantity_interpolate_accumulation(input, reference_point, hb, Az_column_idx, itp_strategy, leaf_idx)
                    ∇Af += ∇AfW; ∇Axb += ∇AxbW; ∇Ayb += ∇AybW; ∇Azb += ∇AzbW
                    mWlρ += _ShepardNormalization_accumulation(input, reference_point, hb, itp_strategy, leaf_idx)
                end
            end
            if RR[node]
                @inbounds leaf_idx = R[node]
                hb = input.h[leaf_idx]
                rleaf = Kvalid * hb; r2leaf = rleaf * rleaf
                d2 = NeighborSearch._dist2_to_leaf_aabb(leaf_min, leaf_max, reference_point, leaf_idx)
                if d2 <= r2leaf
                    ∇AfW, ∇AxbW, ∇AybW, ∇AzbW = _divergence_quantity_accumulation(input, reference_point, hb, Ax_column_idx, Ay_column_idx, Az_column_idx, itp_strategy, leaf_idx)
                    ρ += _density_accumulation(input, reference_point, hb, itp_strategy, leaf_idx)
                    Ax += _quantity_interpolate_accumulation(input, reference_point, hb, Ax_column_idx, itp_strategy, leaf_idx)
                    Ay += _quantity_interpolate_accumulation(input, reference_point, hb, Ay_column_idx, itp_strategy, leaf_idx)
                    Az += _quantity_interpolate_accumulation(input, reference_point, hb, Az_column_idx, itp_strategy, leaf_idx)
                    ∇Af += ∇AfW; ∇Axb += ∇AxbW; ∇Ayb += ∇AybW; ∇Azb += ∇AzbW
                    mWlρ += _ShepardNormalization_accumulation(input, reference_point, hb, itp_strategy, leaf_idx)
                end
            end

            if !LL[node]
                node = L[node]; continue
            end
            if !RR[node]
                node = R[node]; continue
            end
            node = NeighborSearch._next_internal_node(node, L, R, LL, RR, parent)
        else
            node = NeighborSearch._next_internal_node(node, L, R, LL, RR, parent)
        end
    end
    if iszero(ρ)
        return T(NaN)
    end
    Ax /= mWlρ; Ay /= mWlρ; Az /= mWlρ
    ∇Ab = Ax * ∇Axb + Ay * ∇Ayb + Az * ∇Azb
    return (∇Af - ∇Ab)/ρ
end

@inline function _divergence_quantity_interpolate_kernel(input::ITPINPUT, reference_point::NTuple{3, T}, ha :: T, LBVH :: LinearBVH, Ax_column_idx :: Int64, Ay_column_idx :: Int64, Az_column_idx :: Int64, itp_strategy :: Type{itpSymmetric}) where {ITPINPUT <: AbstractInterpolationInput, T <: AbstractFloat}
    Ktyp = typeof(input.smoothed_kernel)
    Kvalid = KernelFunctionValid(Ktyp, T)

    ∇Af = zero(T); ∇Axb = zero(T); ∇Ayb = zero(T); ∇Azb = zero(T)
    mWlρ = zero(T); Ax = zero(T); Ay = zero(T); Az = zero(T); ρ = zero(T)

    node_min = LBVH.node_aabb.min; node_max = LBVH.node_aabb.max
    leaf_min = LBVH.leaf_aabb.min; leaf_max = LBVH.leaf_aabb.max
    node_hmax = LBVH.node_hmax
    L = LBVH.brt.left_child; R = LBVH.brt.right_child
    LL = LBVH.brt.is_leaf_left; RR = LBVH.brt.is_leaf_right
    parent = LBVH.brt.node_parent; root = LBVH.root

    if iszero(root)
        nleaf = length(leaf_min[1])
        @inbounds for leaf_idx in 1:nleaf
            hb = input.h[leaf_idx]
            r = Kvalid * max(ha, hb); r2 = r * r
            d2 = NeighborSearch._dist2_to_leaf_aabb(leaf_min, leaf_max, reference_point, leaf_idx)
            if d2 <= r2
                ∇AfW, ∇AxbW, ∇AybW, ∇AzbW = _divergence_quantity_accumulation(input, reference_point, ha, hb, Ax_column_idx, Ay_column_idx, Az_column_idx, itp_strategy, leaf_idx)
                ρ += _density_accumulation(input, reference_point, ha, hb, itp_strategy, leaf_idx)
                Ax += _quantity_interpolate_accumulation(input, reference_point, ha, hb, Ax_column_idx, itp_strategy, leaf_idx)
                Ay += _quantity_interpolate_accumulation(input, reference_point, ha, hb, Ay_column_idx, itp_strategy, leaf_idx)
                Az += _quantity_interpolate_accumulation(input, reference_point, ha, hb, Az_column_idx, itp_strategy, leaf_idx)
                ∇Af += ∇AfW; ∇Axb += ∇AxbW; ∇Ayb += ∇AybW; ∇Azb += ∇AzbW
                mWlρ += _ShepardNormalization_accumulation(input, reference_point, ha, hb, itp_strategy, leaf_idx)
            end
        end
        if iszero(ρ)
            return T(NaN)
        end
        Ax /= mWlρ; Ay /= mWlρ; Az /= mWlρ
        ∇Ab = Ax * ∇Axb + Ay * ∇Ayb + Az * ∇Azb
        return (∇Af - ∇Ab)/ρ
    end

    node = root
    while node != 0
        rnode = Kvalid * max(ha, node_hmax[node]); r2node = rnode * rnode
        d2node = NeighborSearch._dist2_to_node_aabb(node_min, node_max, reference_point, node)
        if d2node <= r2node
            if LL[node]
                @inbounds leaf_idx = L[node]
                hb = input.h[leaf_idx]
                rleaf = Kvalid * max(ha, hb); r2leaf = rleaf * rleaf
                d2 = NeighborSearch._dist2_to_leaf_aabb(leaf_min, leaf_max, reference_point, leaf_idx)
                if d2 <= r2leaf
                    ∇AfW, ∇AxbW, ∇AybW, ∇AzbW = _divergence_quantity_accumulation(input, reference_point, ha, hb, Ax_column_idx, Ay_column_idx, Az_column_idx, itp_strategy, leaf_idx)
                    ρ += _density_accumulation(input, reference_point, ha, hb, itp_strategy, leaf_idx)
                    Ax += _quantity_interpolate_accumulation(input, reference_point, ha, hb, Ax_column_idx, itp_strategy, leaf_idx)
                    Ay += _quantity_interpolate_accumulation(input, reference_point, ha, hb, Ay_column_idx, itp_strategy, leaf_idx)
                    Az += _quantity_interpolate_accumulation(input, reference_point, ha, hb, Az_column_idx, itp_strategy, leaf_idx)
                    ∇Af += ∇AfW; ∇Axb += ∇AxbW; ∇Ayb += ∇AybW; ∇Azb += ∇AzbW
                    mWlρ += _ShepardNormalization_accumulation(input, reference_point, ha, hb, itp_strategy, leaf_idx)
                end
            end
            if RR[node]
                @inbounds leaf_idx = R[node]
                hb = input.h[leaf_idx]
                rleaf = Kvalid * max(ha, hb); r2leaf = rleaf * rleaf
                d2 = NeighborSearch._dist2_to_leaf_aabb(leaf_min, leaf_max, reference_point, leaf_idx)
                if d2 <= r2leaf
                    ∇AfW, ∇AxbW, ∇AybW, ∇AzbW = _divergence_quantity_accumulation(input, reference_point, ha, hb, Ax_column_idx, Ay_column_idx, Az_column_idx, itp_strategy, leaf_idx)
                    ρ += _density_accumulation(input, reference_point, ha, hb, itp_strategy, leaf_idx)
                    Ax += _quantity_interpolate_accumulation(input, reference_point, ha, hb, Ax_column_idx, itp_strategy, leaf_idx)
                    Ay += _quantity_interpolate_accumulation(input, reference_point, ha, hb, Ay_column_idx, itp_strategy, leaf_idx)
                    Az += _quantity_interpolate_accumulation(input, reference_point, ha, hb, Az_column_idx, itp_strategy, leaf_idx)
                    ∇Af += ∇AfW; ∇Axb += ∇AxbW; ∇Ayb += ∇AybW; ∇Azb += ∇AzbW
                    mWlρ += _ShepardNormalization_accumulation(input, reference_point, ha, hb, itp_strategy, leaf_idx)
                end
            end

            if !LL[node]
                node = L[node]; continue
            end
            if !RR[node]
                node = R[node]; continue
            end
            node = NeighborSearch._next_internal_node(node, L, R, LL, RR, parent)
        else
            node = NeighborSearch._next_internal_node(node, L, R, LL, RR, parent)
        end
    end
    if iszero(ρ)
        return T(NaN)
    end
    Ax /= mWlρ; Ay /= mWlρ; Az /= mWlρ
    ∇Ab = Ax * ∇Axb + Ay * ∇Ayb + Az * ∇Azb
    return (∇Af - ∇Ab)/ρ
end