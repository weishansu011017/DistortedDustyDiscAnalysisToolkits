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