@inline function _gradient_density_kernel(input::ITPINPUT, reference_point::NTuple{3, T}, ha :: T, LBVH :: LinearBVH, itp_strategy :: Type{itpGather} = itpGather) where {ITPINPUT <: AbstractInterpolationInput, T <: AbstractFloat}
    # Prepare for interpolation
    Ktyp = typeof(input.smoothed_kernel)
    Kvalid = KernelFunctionValid(Ktyp, T)

    # Initialize counter
    ∇ρxf :: T = zero(T)
    ∇ρyf :: T = zero(T)
    ∇ρzf :: T = zero(T)
    ∇ρxb :: T = zero(T)
    ∇ρyb :: T = zero(T)
    ∇ρzb :: T = zero(T)

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
                ∇ρxfW, ∇ρyfW, ∇ρzfW, ∇ρxbW, ∇ρybW, ∇ρzbW = _gradient_density_accumulation(input, reference_point, ha, itp_strategy, leaf_idx)
                ρ += _density_accumulation(input, reference_point, ha, itp_strategy, leaf_idx)
                ∇ρxf += ∇ρxfW
                ∇ρyf += ∇ρyfW
                ∇ρzf += ∇ρzfW
                ∇ρxb += ∇ρxbW
                ∇ρyb += ∇ρybW
                ∇ρzb += ∇ρzbW
                #########################################################
            end
        end
        if iszero(ρ)
            return (T(NaN), T(NaN), T(NaN))
        end

        # Construct gradient
        ∇ρxf /= ρ
        ∇ρyf /= ρ
        ∇ρzf /= ρ

        # Final result
        ∇ρx = (∇ρxf - ∇ρxb)
        ∇ρy = (∇ρyf - ∇ρyb)
        ∇ρz = (∇ρzf - ∇ρzb)
        return (∇ρx, ∇ρy, ∇ρz)
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
                    ∇ρxfW, ∇ρyfW, ∇ρzfW, ∇ρxbW, ∇ρybW, ∇ρzbW = _gradient_density_accumulation(input, reference_point, ha, itp_strategy, leaf_idx)
                    ρ += _density_accumulation(input, reference_point, ha, itp_strategy, leaf_idx)
                    ∇ρxf += ∇ρxfW
                    ∇ρyf += ∇ρyfW
                    ∇ρzf += ∇ρzfW
                    ∇ρxb += ∇ρxbW
                    ∇ρyb += ∇ρybW
                    ∇ρzb += ∇ρzbW
                    #########################################################
                end
            end
            if RR[node]
                @inbounds leaf_idx = R[node]
                d2 = NeighborSearch._dist2_to_leaf_aabb(leaf_min, leaf_max, reference_point, leaf_idx)
                if d2 <= radius2
                    ########### Found a neighbor, do accumulation ###########
                    ∇ρxfW, ∇ρyfW, ∇ρzfW, ∇ρxbW, ∇ρybW, ∇ρzbW = _gradient_density_accumulation(input, reference_point, ha, itp_strategy, leaf_idx)
                    ρ += _density_accumulation(input, reference_point, ha, itp_strategy, leaf_idx)
                    ∇ρxf += ∇ρxfW
                    ∇ρyf += ∇ρyfW
                    ∇ρzf += ∇ρzfW
                    ∇ρxb += ∇ρxbW
                    ∇ρyb += ∇ρybW
                    ∇ρzb += ∇ρzbW
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
        return (T(NaN), T(NaN), T(NaN))
    end

    # Construct gradient
    ∇ρxf /= ρ
    ∇ρyf /= ρ
    ∇ρzf /= ρ

    # Final result
    ∇ρx = (∇ρxf - ∇ρxb)
    ∇ρy = (∇ρyf - ∇ρyb)
    ∇ρz = (∇ρzf - ∇ρzb)
    return (∇ρx, ∇ρy, ∇ρz)
end

@inline function _gradient_quantity_interpolate_kernel(input::ITPINPUT, reference_point::NTuple{3, T}, ha :: T, LBVH :: LinearBVH, column_idx :: Int64, itp_strategy :: Type{itpGather} = itpGather) where {ITPINPUT <: AbstractInterpolationInput, T <: AbstractFloat}
    # Prepare for interpolation
    Ktyp = typeof(input.smoothed_kernel)
    Kvalid = KernelFunctionValid(Ktyp, T)

    # Initialize counter
    ∇Axf :: T = zero(T)
    ∇Ayf :: T = zero(T)
    ∇Azf :: T = zero(T)
    ∇Axb :: T = zero(T)
    ∇Ayb :: T = zero(T)
    ∇Azb :: T = zero(T)

    mWlρ :: T = zero(T)
    A :: T = zero(T)
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
                ∇AxfW, ∇AyfW, ∇AzfW, ∇AxbW, ∇AybW, ∇AzbW = _gradient_quantity_accumulation(input, reference_point, ha, column_idx, itp_strategy, leaf_idx)
                ρ += _density_accumulation(input, reference_point, ha, itp_strategy, leaf_idx)
                A += _quantity_interpolate_accumulation(input, reference_point, ha, column_idx, itp_strategy, leaf_idx)
                ∇Axf += ∇AxfW
                ∇Ayf += ∇AyfW
                ∇Azf += ∇AzfW
                ∇Axb += ∇AxbW
                ∇Ayb += ∇AybW
                ∇Azb += ∇AzbW
                mWlρ += _ShepardNormalization_accumulation(input, reference_point, ha, itp_strategy, leaf_idx)
                #########################################################
            end
        end
        if iszero(ρ)
            return (T(NaN), T(NaN), T(NaN))
        end

        # Shepard normalization
        A /= mWlρ

        # Construct gradient
        ∇Axb *= A
        ∇Ayb *= A
        ∇Azb *= A

        # Final result
        ∇Ax = (∇Axf - ∇Axb)/ρ
        ∇Ay = (∇Ayf - ∇Ayb)/ρ
        ∇Az = (∇Azf - ∇Azb)/ρ
        return (∇Ax, ∇Ay, ∇Az)
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
                    ∇AxfW, ∇AyfW, ∇AzfW, ∇AxbW, ∇AybW, ∇AzbW = _gradient_quantity_accumulation(input, reference_point, ha, column_idx, itp_strategy, leaf_idx)
                    ρ += _density_accumulation(input, reference_point, ha, itp_strategy, leaf_idx)
                    A += _quantity_interpolate_accumulation(input, reference_point, ha, column_idx, itp_strategy, leaf_idx)
                    ∇Axf += ∇AxfW
                    ∇Ayf += ∇AyfW
                    ∇Azf += ∇AzfW
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
                    ∇AxfW, ∇AyfW, ∇AzfW, ∇AxbW, ∇AybW, ∇AzbW = _gradient_quantity_accumulation(input, reference_point, ha, column_idx, itp_strategy, leaf_idx)
                    ρ += _density_accumulation(input, reference_point, ha, itp_strategy, leaf_idx)
                    A += _quantity_interpolate_accumulation(input, reference_point, ha, column_idx, itp_strategy, leaf_idx)
                    ∇Axf += ∇AxfW
                    ∇Ayf += ∇AyfW
                    ∇Azf += ∇AzfW
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
        return (T(NaN), T(NaN), T(NaN))
    end

    # Shepard normalization
    A /= mWlρ

    # Construct gradient
    ∇Axb *= A
    ∇Ayb *= A
    ∇Azb *= A

    # Final result
    ∇Ax = (∇Axf - ∇Axb)/ρ
    ∇Ay = (∇Ayf - ∇Ayb)/ρ
    ∇Az = (∇Azf - ∇Azb)/ρ
    return (∇Ax, ∇Ay, ∇Az)
end

@inline function _gradient_density_kernel(input::ITPINPUT, reference_point::NTuple{3, T}, ha :: T, LBVH :: LinearBVH, itp_strategy :: Type{itpScatter}) where {ITPINPUT <: AbstractInterpolationInput, T <: AbstractFloat}
    Ktyp = typeof(input.smoothed_kernel)
    Kvalid = KernelFunctionValid(Ktyp, T)

    ∇ρxf = ∇ρyf = ∇ρzf = zero(T)
    ∇ρxb = ∇ρyb = ∇ρzb = zero(T)
    ρ = zero(T)

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
                ∇ρxfW, ∇ρyfW, ∇ρzfW, ∇ρxbW, ∇ρybW, ∇ρzbW = _gradient_density_accumulation(input, reference_point, hb, itp_strategy, leaf_idx)
                ρ += _density_accumulation(input, reference_point, hb, itp_strategy, leaf_idx)
                ∇ρxf += ∇ρxfW; ∇ρyf += ∇ρyfW; ∇ρzf += ∇ρzfW
                ∇ρxb += ∇ρxbW; ∇ρyb += ∇ρybW; ∇ρzb += ∇ρzbW
            end
        end
        if iszero(ρ)
            return (T(NaN), T(NaN), T(NaN))
        end
        ∇ρxf /= ρ; ∇ρyf /= ρ; ∇ρzf /= ρ
        return (∇ρxf - ∇ρxb, ∇ρyf - ∇ρyb, ∇ρzf - ∇ρzb)
    end

    node = root
    while node != 0
        rnode = Kvalid * node_hmax[node]
        r2node = rnode * rnode
        dist2_node = NeighborSearch._dist2_to_node_aabb(node_min, node_max, reference_point, node)
        if dist2_node <= r2node
            if LL[node]
                @inbounds leaf_idx = L[node]
                hb = input.h[leaf_idx]
                rleaf = Kvalid * hb
                r2leaf = rleaf * rleaf
                d2 = NeighborSearch._dist2_to_leaf_aabb(leaf_min, leaf_max, reference_point, leaf_idx)
                if d2 <= r2leaf
                    ∇ρxfW, ∇ρyfW, ∇ρzfW, ∇ρxbW, ∇ρybW, ∇ρzbW = _gradient_density_accumulation(input, reference_point, hb, itp_strategy, leaf_idx)
                    ρ += _density_accumulation(input, reference_point, hb, itp_strategy, leaf_idx)
                    ∇ρxf += ∇ρxfW; ∇ρyf += ∇ρyfW; ∇ρzf += ∇ρzfW
                    ∇ρxb += ∇ρxbW; ∇ρyb += ∇ρybW; ∇ρzb += ∇ρzbW
                end
            end
            if RR[node]
                @inbounds leaf_idx = R[node]
                hb = input.h[leaf_idx]
                rleaf = Kvalid * hb
                r2leaf = rleaf * rleaf
                d2 = NeighborSearch._dist2_to_leaf_aabb(leaf_min, leaf_max, reference_point, leaf_idx)
                if d2 <= r2leaf
                    ∇ρxfW, ∇ρyfW, ∇ρzfW, ∇ρxbW, ∇ρybW, ∇ρzbW = _gradient_density_accumulation(input, reference_point, hb, itp_strategy, leaf_idx)
                    ρ += _density_accumulation(input, reference_point, hb, itp_strategy, leaf_idx)
                    ∇ρxf += ∇ρxfW; ∇ρyf += ∇ρyfW; ∇ρzf += ∇ρzfW
                    ∇ρxb += ∇ρxbW; ∇ρyb += ∇ρybW; ∇ρzb += ∇ρzbW
                end
            end

            if !LL[node]
                node = L[node]; continue
            end
            if !RR[node]
                node = R[node]; continue
            end
            node = NeighborSearch._next_internal_node(node, L, R, LL, RR, node_parent)
        else
            node = NeighborSearch._next_internal_node(node, L, R, LL, RR, node_parent)
        end
    end
    if iszero(ρ)
        return (T(NaN), T(NaN), T(NaN))
    end
    ∇ρxf /= ρ; ∇ρyf /= ρ; ∇ρzf /= ρ
    return (∇ρxf - ∇ρxb, ∇ρyf - ∇ρyb, ∇ρzf - ∇ρzb)
end

@inline function _gradient_density_kernel(input::ITPINPUT, reference_point::NTuple{3, T}, ha :: T, LBVH :: LinearBVH, itp_strategy :: Type{itpSymmetric}) where {ITPINPUT <: AbstractInterpolationInput, T <: AbstractFloat}
    Ktyp = typeof(input.smoothed_kernel)
    Kvalid = KernelFunctionValid(Ktyp, T)

    ∇ρxf = ∇ρyf = ∇ρzf = zero(T)
    ∇ρxb = ∇ρyb = ∇ρzb = zero(T)
    ρ = zero(T)

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
                ∇ρxfW, ∇ρyfW, ∇ρzfW, ∇ρxbW, ∇ρybW, ∇ρzbW = _gradient_density_accumulation(input, reference_point, ha, hb, itp_strategy, leaf_idx)
                ρ += _density_accumulation(input, reference_point, ha, hb, itp_strategy, leaf_idx)
                ∇ρxf += ∇ρxfW; ∇ρyf += ∇ρyfW; ∇ρzf += ∇ρzfW
                ∇ρxb += ∇ρxbW; ∇ρyb += ∇ρybW; ∇ρzb += ∇ρzbW
            end
        end
        if iszero(ρ)
            return (T(NaN), T(NaN), T(NaN))
        end
        ∇ρxf /= ρ; ∇ρyf /= ρ; ∇ρzf /= ρ
        return (∇ρxf - ∇ρxb, ∇ρyf - ∇ρyb, ∇ρzf - ∇ρzb)
    end

    node = root
    while node != 0
        rnode = Kvalid * max(ha, node_hmax[node])
        r2node = rnode * rnode
        dist2_node = NeighborSearch._dist2_to_node_aabb(node_min, node_max, reference_point, node)
        if dist2_node <= r2node
            if LL[node]
                @inbounds leaf_idx = L[node]
                hb = input.h[leaf_idx]
                rleaf = Kvalid * max(ha, hb)
                r2leaf = rleaf * rleaf
                d2 = NeighborSearch._dist2_to_leaf_aabb(leaf_min, leaf_max, reference_point, leaf_idx)
                if d2 <= r2leaf
                    ∇ρxfW, ∇ρyfW, ∇ρzfW, ∇ρxbW, ∇ρybW, ∇ρzbW = _gradient_density_accumulation(input, reference_point, ha, hb, itp_strategy, leaf_idx)
                    ρ += _density_accumulation(input, reference_point, ha, hb, itp_strategy, leaf_idx)
                    ∇ρxf += ∇ρxfW; ∇ρyf += ∇ρyfW; ∇ρzf += ∇ρzfW
                    ∇ρxb += ∇ρxbW; ∇ρyb += ∇ρybW; ∇ρzb += ∇ρzbW
                end
            end
            if RR[node]
                @inbounds leaf_idx = R[node]
                hb = input.h[leaf_idx]
                rleaf = Kvalid * max(ha, hb)
                r2leaf = rleaf * rleaf
                d2 = NeighborSearch._dist2_to_leaf_aabb(leaf_min, leaf_max, reference_point, leaf_idx)
                if d2 <= r2leaf
                    ∇ρxfW, ∇ρyfW, ∇ρzfW, ∇ρxbW, ∇ρybW, ∇ρzbW = _gradient_density_accumulation(input, reference_point, ha, hb, itp_strategy, leaf_idx)
                    ρ += _density_accumulation(input, reference_point, ha, hb, itp_strategy, leaf_idx)
                    ∇ρxf += ∇ρxfW; ∇ρyf += ∇ρyfW; ∇ρzf += ∇ρzfW
                    ∇ρxb += ∇ρxbW; ∇ρyb += ∇ρybW; ∇ρzb += ∇ρzbW
                end
            end

            if !LL[node]
                node = L[node]; continue
            end
            if !RR[node]
                node = R[node]; continue
            end
            node = NeighborSearch._next_internal_node(node, L, R, LL, RR, node_parent)
        else
            node = NeighborSearch._next_internal_node(node, L, R, LL, RR, node_parent)
        end
    end
    if iszero(ρ)
        return (T(NaN), T(NaN), T(NaN))
    end
    ∇ρxf /= ρ; ∇ρyf /= ρ; ∇ρzf /= ρ
    return (∇ρxf - ∇ρxb, ∇ρyf - ∇ρyb, ∇ρzf - ∇ρzb)
end

@inline function _gradient_quantity_interpolate_kernel(input::ITPINPUT, reference_point::NTuple{3, T}, ha :: T, LBVH :: LinearBVH, column_idx :: Int64, itp_strategy :: Type{itpScatter}) where {ITPINPUT <: AbstractInterpolationInput, T <: AbstractFloat}
    Ktyp = typeof(input.smoothed_kernel)
    Kvalid = KernelFunctionValid(Ktyp, T)

    ∇Axf = ∇Ayf = ∇Azf = zero(T)
    ∇Axb = ∇Ayb = ∇Azb = zero(T)
    mWlρ = zero(T)
    A = zero(T)
    ρ = zero(T)

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
                ∇AxfW, ∇AyfW, ∇AzfW, ∇AxbW, ∇AybW, ∇AzbW = _gradient_quantity_accumulation(input, reference_point, hb, column_idx, itp_strategy, leaf_idx)
                ρ += _density_accumulation(input, reference_point, hb, itp_strategy, leaf_idx)
                A += _quantity_interpolate_accumulation(input, reference_point, hb, column_idx, itp_strategy, leaf_idx)
                ∇Axf += ∇AxfW; ∇Ayf += ∇AyfW; ∇Azf += ∇AzfW
                ∇Axb += ∇AxbW; ∇Ayb += ∇AybW; ∇Azb += ∇AzbW
                mWlρ += _ShepardNormalization_accumulation(input, reference_point, hb, itp_strategy, leaf_idx)
            end
        end
        if iszero(ρ)
            return (T(NaN), T(NaN), T(NaN))
        end
        A /= mWlρ
        ∇Axb *= A; ∇Ayb *= A; ∇Azb *= A
        return ((∇Axf - ∇Axb)/ρ, (∇Ayf - ∇Ayb)/ρ, (∇Azf - ∇Azb)/ρ)
    end

    node = root
    while node != 0
        rnode = Kvalid * node_hmax[node]
        r2node = rnode * rnode
        dist2_node = NeighborSearch._dist2_to_node_aabb(node_min, node_max, reference_point, node)
        if dist2_node <= r2node
            if LL[node]
                @inbounds leaf_idx = L[node]
                hb = input.h[leaf_idx]
                rleaf = Kvalid * hb
                r2leaf = rleaf * rleaf
                d2 = NeighborSearch._dist2_to_leaf_aabb(leaf_min, leaf_max, reference_point, leaf_idx)
                if d2 <= r2leaf
                    ∇AxfW, ∇AyfW, ∇AzfW, ∇AxbW, ∇AybW, ∇AzbW = _gradient_quantity_accumulation(input, reference_point, hb, column_idx, itp_strategy, leaf_idx)
                    ρ += _density_accumulation(input, reference_point, hb, itp_strategy, leaf_idx)
                    A += _quantity_interpolate_accumulation(input, reference_point, hb, column_idx, itp_strategy, leaf_idx)
                    ∇Axf += ∇AxfW; ∇Ayf += ∇AyfW; ∇Azf += ∇AzfW
                    ∇Axb += ∇AxbW; ∇Ayb += ∇AybW; ∇Azb += ∇AzbW
                    mWlρ += _ShepardNormalization_accumulation(input, reference_point, hb, itp_strategy, leaf_idx)
                end
            end
            if RR[node]
                @inbounds leaf_idx = R[node]
                hb = input.h[leaf_idx]
                rleaf = Kvalid * hb
                r2leaf = rleaf * rleaf
                d2 = NeighborSearch._dist2_to_leaf_aabb(leaf_min, leaf_max, reference_point, leaf_idx)
                if d2 <= r2leaf
                    ∇AxfW, ∇AyfW, ∇AzfW, ∇AxbW, ∇AybW, ∇AzbW = _gradient_quantity_accumulation(input, reference_point, hb, column_idx, itp_strategy, leaf_idx)
                    ρ += _density_accumulation(input, reference_point, hb, itp_strategy, leaf_idx)
                    A += _quantity_interpolate_accumulation(input, reference_point, hb, column_idx, itp_strategy, leaf_idx)
                    ∇Axf += ∇AxfW; ∇Ayf += ∇AyfW; ∇Azf += ∇AzfW
                    ∇Axb += ∇AxbW; ∇Ayb += ∇AybW; ∇Azb += ∇AzbW
                    mWlρ += _ShepardNormalization_accumulation(input, reference_point, hb, itp_strategy, leaf_idx)
                end
            end

            if !LL[node]
                node = L[node]; continue
            end
            if !RR[node]
                node = R[node]; continue
            end
            node = NeighborSearch._next_internal_node(node, L, R, LL, RR, node_parent)
        else
            node = NeighborSearch._next_internal_node(node, L, R, LL, RR, node_parent)
        end
    end
    if iszero(ρ)
        return (T(NaN), T(NaN), T(NaN))
    end
    A /= mWlρ
    ∇Axb *= A; ∇Ayb *= A; ∇Azb *= A
    return ((∇Axf - ∇Axb)/ρ, (∇Ayf - ∇Ayb)/ρ, (∇Azf - ∇Azb)/ρ)
end

@inline function _gradient_quantity_interpolate_kernel(input::ITPINPUT, reference_point::NTuple{3, T}, ha :: T, LBVH :: LinearBVH, column_idx :: Int64, itp_strategy :: Type{itpSymmetric}) where {ITPINPUT <: AbstractInterpolationInput, T <: AbstractFloat}
    Ktyp = typeof(input.smoothed_kernel)
    Kvalid = KernelFunctionValid(Ktyp, T)

    ∇Axf = ∇Ayf = ∇Azf = zero(T)
    ∇Axb = ∇Ayb = ∇Azb = zero(T)
    mWlρ = zero(T)
    A = zero(T)
    ρ = zero(T)

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
                ∇AxfW, ∇AyfW, ∇AzfW, ∇AxbW, ∇AybW, ∇AzbW = _gradient_quantity_accumulation(input, reference_point, ha, hb, column_idx, itp_strategy, leaf_idx)
                ρ += _density_accumulation(input, reference_point, ha, hb, itp_strategy, leaf_idx)
                A += _quantity_interpolate_accumulation(input, reference_point, ha, hb, column_idx, itp_strategy, leaf_idx)
                ∇Axf += ∇AxfW; ∇Ayf += ∇AyfW; ∇Azf += ∇AzfW
                ∇Axb += ∇AxbW; ∇Ayb += ∇AybW; ∇Azb += ∇AzbW
                mWlρ += _ShepardNormalization_accumulation(input, reference_point, ha, hb, itp_strategy, leaf_idx)
            end
        end
        if iszero(ρ)
            return (T(NaN), T(NaN), T(NaN))
        end
        A /= mWlρ
        ∇Axb *= A; ∇Ayb *= A; ∇Azb *= A
        return ((∇Axf - ∇Axb)/ρ, (∇Ayf - ∇Ayb)/ρ, (∇Azf - ∇Azb)/ρ)
    end

    node = root
    while node != 0
        rnode = Kvalid * max(ha, node_hmax[node])
        r2node = rnode * rnode
        dist2_node = NeighborSearch._dist2_to_node_aabb(node_min, node_max, reference_point, node)
        if dist2_node <= r2node
            if LL[node]
                @inbounds leaf_idx = L[node]
                hb = input.h[leaf_idx]
                rleaf = Kvalid * max(ha, hb)
                r2leaf = rleaf * rleaf
                d2 = NeighborSearch._dist2_to_leaf_aabb(leaf_min, leaf_max, reference_point, leaf_idx)
                if d2 <= r2leaf
                    ∇AxfW, ∇AyfW, ∇AzfW, ∇AxbW, ∇AybW, ∇AzbW = _gradient_quantity_accumulation(input, reference_point, ha, hb, column_idx, itp_strategy, leaf_idx)
                    ρ += _density_accumulation(input, reference_point, ha, hb, itp_strategy, leaf_idx)
                    A += _quantity_interpolate_accumulation(input, reference_point, ha, hb, column_idx, itp_strategy, leaf_idx)
                    ∇Axf += ∇AxfW; ∇Ayf += ∇AyfW; ∇Azf += ∇AzfW
                    ∇Axb += ∇AxbW; ∇Ayb += ∇AybW; ∇Azb += ∇AzbW
                    mWlρ += _ShepardNormalization_accumulation(input, reference_point, ha, hb, itp_strategy, leaf_idx)
                end
            end
            if RR[node]
                @inbounds leaf_idx = R[node]
                hb = input.h[leaf_idx]
                rleaf = Kvalid * max(ha, hb)
                r2leaf = rleaf * rleaf
                d2 = NeighborSearch._dist2_to_leaf_aabb(leaf_min, leaf_max, reference_point, leaf_idx)
                if d2 <= r2leaf
                    ∇AxfW, ∇AyfW, ∇AzfW, ∇AxbW, ∇AybW, ∇AzbW = _gradient_quantity_accumulation(input, reference_point, ha, hb, column_idx, itp_strategy, leaf_idx)
                    ρ += _density_accumulation(input, reference_point, ha, hb, itp_strategy, leaf_idx)
                    A += _quantity_interpolate_accumulation(input, reference_point, ha, hb, column_idx, itp_strategy, leaf_idx)
                    ∇Axf += ∇AxfW; ∇Ayf += ∇AyfW; ∇Azf += ∇AzfW
                    ∇Axb += ∇AxbW; ∇Ayb += ∇AybW; ∇Azb += ∇AzbW
                    mWlρ += _ShepardNormalization_accumulation(input, reference_point, ha, hb, itp_strategy, leaf_idx)
                end
            end

            if !LL[node]
                node = L[node]; continue
            end
            if !RR[node]
                node = R[node]; continue
            end
            node = NeighborSearch._next_internal_node(node, L, R, LL, RR, node_parent)
        else
            node = NeighborSearch._next_internal_node(node, L, R, LL, RR, node_parent)
        end
    end
    if iszero(ρ)
        return (T(NaN), T(NaN), T(NaN))
    end
    A /= mWlρ
    ∇Axb *= A; ∇Ayb *= A; ∇Azb *= A
    return ((∇Axf - ∇Axb)/ρ, (∇Ayf - ∇Ayb)/ρ, (∇Azf - ∇Azb)/ρ)
end