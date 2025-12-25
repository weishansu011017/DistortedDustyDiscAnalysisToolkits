@inline function _gradient_density_kernel(input::ITPINPUT, reference_point::NTuple{3, T}, ha :: T, LBVH :: LinearBVH, :: Type{itpGather}) where {ITPINPUT <: AbstractInterpolationInput, T <: AbstractFloat}
    # Prepare for interpolation
    K = input.smoothed_kernel
    Ktyp = typeof(K)
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
                @inbounds begin
                    rb = (input.x[leaf_idx], input.y[leaf_idx], input.z[leaf_idx])
                    mb = input.m[leaf_idx]
                    ρb = input.ρ[leaf_idx]

                    ∇ρxfW, ∇ρyfW, ∇ρzfW, ∇ρxbW, ∇ρybW, ∇ρzbW = _gradient_density_accumulation(reference_point, rb, mb, ρb, ha, K)
                    ρ += _density_accumulation(reference_point, rb, mb, ha, K)
                    ∇ρxf += ∇ρxfW
                    ∇ρyf += ∇ρyfW
                    ∇ρzf += ∇ρzfW
                    ∇ρxb += ∇ρxbW
                    ∇ρyb += ∇ρybW
                    ∇ρzb += ∇ρzbW
                end
                #########################################################
            end
        end
        if iszero(ρ)
            return (T(NaN), T(NaN), T(NaN))
        end

        # Construct gradient
        ∇ρxb *= ρ
        ∇ρyb *= ρ
        ∇ρzb *= ρ

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
                    @inbounds begin
                        rb = (input.x[leaf_idx], input.y[leaf_idx], input.z[leaf_idx])
                        mb = input.m[leaf_idx]
                        ρb = input.ρ[leaf_idx]

                        ∇ρxfW, ∇ρyfW, ∇ρzfW, ∇ρxbW, ∇ρybW, ∇ρzbW = _gradient_density_accumulation(reference_point, rb, mb, ρb, ha, K)
                        ρ += _density_accumulation(reference_point, rb, mb, ha, K)
                        ∇ρxf += ∇ρxfW
                        ∇ρyf += ∇ρyfW
                        ∇ρzf += ∇ρzfW
                        ∇ρxb += ∇ρxbW
                        ∇ρyb += ∇ρybW
                        ∇ρzb += ∇ρzbW
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
                        rb = (input.x[leaf_idx], input.y[leaf_idx], input.z[leaf_idx])
                        mb = input.m[leaf_idx]
                        ρb = input.ρ[leaf_idx]

                        ∇ρxfW, ∇ρyfW, ∇ρzfW, ∇ρxbW, ∇ρybW, ∇ρzbW = _gradient_density_accumulation(reference_point, rb, mb, ρb, ha, K)
                        ρ += _density_accumulation(reference_point, rb, mb, ha, K)
                        ∇ρxf += ∇ρxfW
                        ∇ρyf += ∇ρyfW
                        ∇ρzf += ∇ρzfW
                        ∇ρxb += ∇ρxbW
                        ∇ρyb += ∇ρybW
                        ∇ρzb += ∇ρzbW
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
    if iszero(ρ)
        return (T(NaN), T(NaN), T(NaN))
    end

    # Construct gradient
    ∇ρxb *= ρ
    ∇ρyb *= ρ
    ∇ρzb *= ρ

    # Final result
    ∇ρx = (∇ρxf - ∇ρxb)
    ∇ρy = (∇ρyf - ∇ρyb)
    ∇ρz = (∇ρzf - ∇ρzb)
    return (∇ρx, ∇ρy, ∇ρz)
end

@inline function _gradient_density_kernel(input::ITPINPUT, reference_point::NTuple{3, T}, ha :: T, LBVH :: LinearBVH, :: Type{itpScatter}) where {ITPINPUT <: AbstractInterpolationInput, T <: AbstractFloat}
    # Prepare for interpolation
    K = input.smoothed_kernel
    Ktyp = typeof(K)
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
    node_hmax = LBVH.node_hmax

    L  = LBVH.brt.left_child
    R  = LBVH.brt.right_child
    LL = LBVH.brt.is_leaf_left
    RR = LBVH.brt.is_leaf_right
    node_parent = LBVH.brt.node_parent
    root = LBVH.root

    # Handle empty tree
    if iszero(root)
        nleaf = length(leaf_min[1])
        @inbounds for leaf_idx in 1:nleaf
            hb = input.h[leaf_idx]
            radius = Kvalid * hb
            radius2 = radius * radius
            d2 = NeighborSearch._dist2_to_leaf_aabb(leaf_min, leaf_max, reference_point, leaf_idx)
            if d2 <= radius2
                ########### Found a neighbor, do accumulation ###########
                @inbounds begin
                    rb = (input.x[leaf_idx], input.y[leaf_idx], input.z[leaf_idx])
                    mb = input.m[leaf_idx]
                    ρb = input.ρ[leaf_idx]

                    ∇ρxfW, ∇ρyfW, ∇ρzfW, ∇ρxbW, ∇ρybW, ∇ρzbW = _gradient_density_accumulation(reference_point, rb, mb, ρb, hb, K)
                    ρ += _density_accumulation(reference_point, rb, mb, hb, K)
                    ∇ρxf += ∇ρxfW
                    ∇ρyf += ∇ρyfW
                    ∇ρzf += ∇ρzfW
                    ∇ρxb += ∇ρxbW
                    ∇ρyb += ∇ρybW
                    ∇ρzb += ∇ρzbW
                end
                #########################################################
            end
        end
        if iszero(ρ)
            return (T(NaN), T(NaN), T(NaN))
        end

        # Construct gradient
        ∇ρxb *= ρ
        ∇ρyb *= ρ
        ∇ρzb *= ρ

        # Final result
        ∇ρx = (∇ρxf - ∇ρxb)
        ∇ρy = (∇ρyf - ∇ρyb)
        ∇ρz = (∇ρzf - ∇ρzb)
        return (∇ρx, ∇ρy, ∇ρz)
    end

    # Start traversal
    node = root
    while node != 0
        rnode = Kvalid * node_hmax[node]
        r2node = rnode * rnode
        dist2_node = NeighborSearch._dist2_to_node_aabb(node_min, node_max, reference_point, node)
        if dist2_node <= r2node
            if LL[node]
                @inbounds leaf_idx = L[node]
                hb = input.h[leaf_idx]
                radius = Kvalid * hb
                radius2 = radius * radius
                d2 = NeighborSearch._dist2_to_leaf_aabb(leaf_min, leaf_max, reference_point, leaf_idx)
                if d2 <= radius2
                    ########### Found a neighbor, do accumulation ###########
                    @inbounds begin
                        rb = (input.x[leaf_idx], input.y[leaf_idx], input.z[leaf_idx])
                        mb = input.m[leaf_idx]
                        ρb = input.ρ[leaf_idx]

                        ∇ρxfW, ∇ρyfW, ∇ρzfW, ∇ρxbW, ∇ρybW, ∇ρzbW = _gradient_density_accumulation(reference_point, rb, mb, ρb, hb, K)
                        ρ += _density_accumulation(reference_point, rb, mb, hb, K)
                        ∇ρxf += ∇ρxfW
                        ∇ρyf += ∇ρyfW
                        ∇ρzf += ∇ρzfW
                        ∇ρxb += ∇ρxbW
                        ∇ρyb += ∇ρybW
                        ∇ρzb += ∇ρzbW
                    end
                    #########################################################
                end
            end
            if RR[node]
                @inbounds leaf_idx = R[node]
                hb = input.h[leaf_idx]
                radius = Kvalid * hb
                radius2 = radius * radius
                d2 = NeighborSearch._dist2_to_leaf_aabb(leaf_min, leaf_max, reference_point, leaf_idx)
                if d2 <= radius2
                    ########### Found a neighbor, do accumulation ###########
                    @inbounds begin
                        rb = (input.x[leaf_idx], input.y[leaf_idx], input.z[leaf_idx])
                        mb = input.m[leaf_idx]
                        ρb = input.ρ[leaf_idx]

                        ∇ρxfW, ∇ρyfW, ∇ρzfW, ∇ρxbW, ∇ρybW, ∇ρzbW = _gradient_density_accumulation(reference_point, rb, mb, ρb, hb, K)
                        ρ += _density_accumulation(reference_point, rb, mb, hb, K)
                        ∇ρxf += ∇ρxfW
                        ∇ρyf += ∇ρyfW
                        ∇ρzf += ∇ρzfW
                        ∇ρxb += ∇ρxbW
                        ∇ρyb += ∇ρybW
                        ∇ρzb += ∇ρzbW
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
    if iszero(ρ)
        return (T(NaN), T(NaN), T(NaN))
    end

    # Construct gradient
    ∇ρxb *= ρ
    ∇ρyb *= ρ
    ∇ρzb *= ρ

    # Final result
    ∇ρx = (∇ρxf - ∇ρxb)
    ∇ρy = (∇ρyf - ∇ρyb)
    ∇ρz = (∇ρzf - ∇ρzb)
    return (∇ρx, ∇ρy, ∇ρz)
end

@inline function _gradient_density_kernel(input::ITPINPUT, reference_point::NTuple{3, T}, ha :: T, LBVH :: LinearBVH, :: Type{itpSymmetric}) where {ITPINPUT <: AbstractInterpolationInput, T <: AbstractFloat}
    # Prepare for interpolation
    K = input.smoothed_kernel
    Ktyp = typeof(K)
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
    node_hmax = LBVH.node_hmax

    L  = LBVH.brt.left_child
    R  = LBVH.brt.right_child
    LL = LBVH.brt.is_leaf_left
    RR = LBVH.brt.is_leaf_right
    node_parent = LBVH.brt.node_parent
    root = LBVH.root

    # Handle empty tree
    if iszero(root)
        nleaf = length(leaf_min[1])
        @inbounds for leaf_idx in 1:nleaf
            hb = input.h[leaf_idx]
            radius = Kvalid * max(ha, hb)
            radius2 = radius * radius
            d2 = NeighborSearch._dist2_to_leaf_aabb(leaf_min, leaf_max, reference_point, leaf_idx)
            if d2 <= radius2
                ########### Found a neighbor, do accumulation ###########
                @inbounds begin
                    rb = (input.x[leaf_idx], input.y[leaf_idx], input.z[leaf_idx])
                    mb = input.m[leaf_idx]
                    ρb = input.ρ[leaf_idx]

                    ∇ρxfW, ∇ρyfW, ∇ρzfW, ∇ρxbW, ∇ρybW, ∇ρzbW = _gradient_density_accumulation(reference_point, rb, mb, ρb, ha, hb, K)
                    ρ += _density_accumulation(reference_point, rb, mb, ha, hb, K)
                    ∇ρxf += ∇ρxfW
                    ∇ρyf += ∇ρyfW
                    ∇ρzf += ∇ρzfW
                    ∇ρxb += ∇ρxbW
                    ∇ρyb += ∇ρybW
                    ∇ρzb += ∇ρzbW
                end
                #########################################################
            end
        end
        if iszero(ρ)
            return (T(NaN), T(NaN), T(NaN))
        end

        # Construct gradient
        ∇ρxb *= ρ
        ∇ρyb *= ρ
        ∇ρzb *= ρ

        # Final result
        ∇ρx = (∇ρxf - ∇ρxb)
        ∇ρy = (∇ρyf - ∇ρyb)
        ∇ρz = (∇ρzf - ∇ρzb)
        return (∇ρx, ∇ρy, ∇ρz)
    end

    # Start traversal
    node = root
    while node != 0
        rnode = Kvalid * max(ha, node_hmax[node])
        r2node = rnode * rnode
        dist2_node = NeighborSearch._dist2_to_node_aabb(node_min, node_max, reference_point, node)
        if dist2_node <= r2node
            if LL[node]
                @inbounds leaf_idx = L[node]
                hb = input.h[leaf_idx]
                radius = Kvalid * max(ha, hb)
                radius2 = radius * radius
                d2 = NeighborSearch._dist2_to_leaf_aabb(leaf_min, leaf_max, reference_point, leaf_idx)
                if d2 <= radius2
                    ########### Found a neighbor, do accumulation ###########
                    @inbounds begin
                        rb = (input.x[leaf_idx], input.y[leaf_idx], input.z[leaf_idx])
                        mb = input.m[leaf_idx]
                        ρb = input.ρ[leaf_idx]

                        ∇ρxfW, ∇ρyfW, ∇ρzfW, ∇ρxbW, ∇ρybW, ∇ρzbW = _gradient_density_accumulation(reference_point, rb, mb, ρb, ha, hb, K)
                        ρ += _density_accumulation(reference_point, rb, mb, ha, hb, K)
                        ∇ρxf += ∇ρxfW
                        ∇ρyf += ∇ρyfW
                        ∇ρzf += ∇ρzfW
                        ∇ρxb += ∇ρxbW
                        ∇ρyb += ∇ρybW
                        ∇ρzb += ∇ρzbW
                    end
                    #########################################################
                end
            end
            if RR[node]
                @inbounds leaf_idx = R[node]
                hb = input.h[leaf_idx]
                radius = Kvalid * max(ha, hb)
                radius2 = radius * radius
                d2 = NeighborSearch._dist2_to_leaf_aabb(leaf_min, leaf_max, reference_point, leaf_idx)
                if d2 <= radius2
                    ########### Found a neighbor, do accumulation ###########
                    @inbounds begin
                        rb = (input.x[leaf_idx], input.y[leaf_idx], input.z[leaf_idx])
                        mb = input.m[leaf_idx]
                        ρb = input.ρ[leaf_idx]

                        ∇ρxfW, ∇ρyfW, ∇ρzfW, ∇ρxbW, ∇ρybW, ∇ρzbW = _gradient_density_accumulation(reference_point, rb, mb, ρb, ha, hb, K)
                        ρ += _density_accumulation(reference_point, rb, mb, ha, hb, K)
                        ∇ρxf += ∇ρxfW
                        ∇ρyf += ∇ρyfW
                        ∇ρzf += ∇ρzfW
                        ∇ρxb += ∇ρxbW
                        ∇ρyb += ∇ρybW
                        ∇ρzb += ∇ρzbW
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
    if iszero(ρ)
        return (T(NaN), T(NaN), T(NaN))
    end

    # Construct gradient
    ∇ρxb *= ρ
    ∇ρyb *= ρ
    ∇ρzb *= ρ

    # Final result
    ∇ρx = (∇ρxf - ∇ρxb)
    ∇ρy = (∇ρyf - ∇ρyb)
    ∇ρz = (∇ρzf - ∇ρzb)
    return (∇ρx, ∇ρy, ∇ρz)
end

@inline function _gradient_quantity_interpolate_kernel(input::ITPINPUT, reference_point::NTuple{3, T}, ha :: T, LBVH :: LinearBVH, column_idx :: Int, :: Type{itpGather}) where {ITPINPUT <: AbstractInterpolationInput, T <: AbstractFloat}
    # Prepare for interpolation
    K = input.smoothed_kernel
    Ktyp = typeof(K)
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
                @inbounds begin
                    rb = (input.x[leaf_idx], input.y[leaf_idx], input.z[leaf_idx])
                    mb = input.m[leaf_idx]
                    ρb = input.ρ[leaf_idx]
                    Ab = input.quant[column_idx][leaf_idx]

                    ∇AxfW, ∇AyfW, ∇AzfW, ∇AxbW, ∇AybW, ∇AzbW = _gradient_quantity_accumulation(reference_point, rb, mb, ρb, Ab, ha, K)
                    ∇Axf += ∇AxfW
                    ∇Ayf += ∇AyfW
                    ∇Azf += ∇AzfW
                    ∇Axb += ∇AxbW
                    ∇Ayb += ∇AybW
                    ∇Azb += ∇AzbW
                    A += _quantity_interpolate_accumulation(reference_point, rb, mb, ρb, Ab, ha, K)
                    mWlρ += _ShepardNormalization_accumulation(reference_point, rb, mb, ρb, ha, K)
                end
                #########################################################
            end
        end
        if iszero(mWlρ)
            return (T(NaN), T(NaN), T(NaN))
        end

        # Shepard normalization
        A /= mWlρ

        # Construct gradient
        ∇Axb *= A
        ∇Ayb *= A
        ∇Azb *= A

        # Final result
        ∇Ax = (∇Axf - ∇Axb)
        ∇Ay = (∇Ayf - ∇Ayb)
        ∇Az = (∇Azf - ∇Azb)
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
                    @inbounds begin
                        rb = (input.x[leaf_idx], input.y[leaf_idx], input.z[leaf_idx])
                        mb = input.m[leaf_idx]
                        ρb = input.ρ[leaf_idx]
                        Ab = input.quant[column_idx][leaf_idx]

                        ∇AxfW, ∇AyfW, ∇AzfW, ∇AxbW, ∇AybW, ∇AzbW = _gradient_quantity_accumulation(reference_point, rb, mb, ρb, Ab, ha, K)
                        ∇Axf += ∇AxfW
                        ∇Ayf += ∇AyfW
                        ∇Azf += ∇AzfW
                        ∇Axb += ∇AxbW
                        ∇Ayb += ∇AybW
                        ∇Azb += ∇AzbW
                        A += _quantity_interpolate_accumulation(reference_point, rb, mb, ρb, Ab, ha, K)
                        mWlρ += _ShepardNormalization_accumulation(reference_point, rb, mb, ρb, ha, K)
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
                        rb = (input.x[leaf_idx], input.y[leaf_idx], input.z[leaf_idx])
                        mb = input.m[leaf_idx]
                        ρb = input.ρ[leaf_idx]
                        Ab = input.quant[column_idx][leaf_idx]

                        ∇AxfW, ∇AyfW, ∇AzfW, ∇AxbW, ∇AybW, ∇AzbW = _gradient_quantity_accumulation(reference_point, rb, mb, ρb, Ab, ha, K)
                        ∇Axf += ∇AxfW
                        ∇Ayf += ∇AyfW
                        ∇Azf += ∇AzfW
                        ∇Axb += ∇AxbW
                        ∇Ayb += ∇AybW
                        ∇Azb += ∇AzbW
                        A += _quantity_interpolate_accumulation(reference_point, rb, mb, ρb, Ab, ha, K)
                        mWlρ += _ShepardNormalization_accumulation(reference_point, rb, mb, ρb, ha, K)
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
    if iszero(mWlρ)
        return (T(NaN), T(NaN), T(NaN))
    end

    # Shepard normalization
    A /= mWlρ

    # Construct gradient
    ∇Axb *= A
    ∇Ayb *= A
    ∇Azb *= A

    # Final result
    ∇Ax = (∇Axf - ∇Axb)
    ∇Ay = (∇Ayf - ∇Ayb)
    ∇Az = (∇Azf - ∇Azb)
    return (∇Ax, ∇Ay, ∇Az)
end

@inline function _gradient_quantity_interpolate_kernel(input::ITPINPUT, reference_point::NTuple{3, T}, ha :: T, LBVH :: LinearBVH, column_idx :: Int, :: Type{itpScatter}) where {ITPINPUT <: AbstractInterpolationInput, T <: AbstractFloat}
    # Prepare for interpolation
    K = input.smoothed_kernel
    Ktyp = typeof(K)
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

    # LBVH data
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

    # Handle empty tree
    if iszero(root)
        nleaf = length(leaf_min[1])
        @inbounds for leaf_idx in 1:nleaf
            hb = input.h[leaf_idx]
            radius = Kvalid * hb
            radius2 = radius * radius
            d2 = NeighborSearch._dist2_to_leaf_aabb(leaf_min, leaf_max, reference_point, leaf_idx)
            if d2 <= radius2
                ########### Found a neighbor, do accumulation ###########
                @inbounds begin
                    rb = (input.x[leaf_idx], input.y[leaf_idx], input.z[leaf_idx])
                    mb = input.m[leaf_idx]
                    ρb = input.ρ[leaf_idx]
                    Ab = input.quant[column_idx][leaf_idx]

                    ∇AxfW, ∇AyfW, ∇AzfW, ∇AxbW, ∇AybW, ∇AzbW = _gradient_quantity_accumulation(reference_point, rb, mb, ρb, Ab, hb, K)
                    ∇Axf += ∇AxfW
                    ∇Ayf += ∇AyfW
                    ∇Azf += ∇AzfW
                    ∇Axb += ∇AxbW
                    ∇Ayb += ∇AybW
                    ∇Azb += ∇AzbW
                    A += _quantity_interpolate_accumulation(reference_point, rb, mb, ρb, Ab, hb, K)
                    mWlρ += _ShepardNormalization_accumulation(reference_point, rb, mb, ρb, hb, K)
                end
                #########################################################
            end
        end
        if iszero(mWlρ)
            return (T(NaN), T(NaN), T(NaN))
        end

        # Shepard normalization
        A /= mWlρ

        # Construct gradient
        ∇Axb *= A
        ∇Ayb *= A
        ∇Azb *= A

        # Final result
        ∇Ax = (∇Axf - ∇Axb)
        ∇Ay = (∇Ayf - ∇Ayb)
        ∇Az = (∇Azf - ∇Azb)
        return (∇Ax, ∇Ay, ∇Az)
    end

    # Start traversal
    node = root
    while node != 0
        rnode = Kvalid * node_hmax[node]
        r2node = rnode * rnode
        dist2_node = NeighborSearch._dist2_to_node_aabb(node_min, node_max, reference_point, node)
        if dist2_node <= r2node
            if LL[node]
                @inbounds leaf_idx = L[node]
                hb = input.h[leaf_idx]
                radius = Kvalid * hb
                radius2 = radius * radius
                d2 = NeighborSearch._dist2_to_leaf_aabb(leaf_min, leaf_max, reference_point, leaf_idx)
                if d2 <= radius2
                    ########### Found a neighbor, do accumulation ###########
                    @inbounds begin
                        rb = (input.x[leaf_idx], input.y[leaf_idx], input.z[leaf_idx])
                        mb = input.m[leaf_idx]
                        ρb = input.ρ[leaf_idx]
                        Ab = input.quant[column_idx][leaf_idx]

                        ∇AxfW, ∇AyfW, ∇AzfW, ∇AxbW, ∇AybW, ∇AzbW = _gradient_quantity_accumulation(reference_point, rb, mb, ρb, Ab, hb, K)
                        ∇Axf += ∇AxfW
                        ∇Ayf += ∇AyfW
                        ∇Azf += ∇AzfW
                        ∇Axb += ∇AxbW
                        ∇Ayb += ∇AybW
                        ∇Azb += ∇AzbW
                        A += _quantity_interpolate_accumulation(reference_point, rb, mb, ρb, Ab, hb, K)
                        mWlρ += _ShepardNormalization_accumulation(reference_point, rb, mb, ρb, hb, K)
                    end
                    #########################################################
                end
            end
            if RR[node]
                @inbounds leaf_idx = R[node]
                hb = input.h[leaf_idx]
                radius = Kvalid * hb
                radius2 = radius * radius
                d2 = NeighborSearch._dist2_to_leaf_aabb(leaf_min, leaf_max, reference_point, leaf_idx)
                if d2 <= radius2
                    ########### Found a neighbor, do accumulation ###########
                    @inbounds begin
                        rb = (input.x[leaf_idx], input.y[leaf_idx], input.z[leaf_idx])
                        mb = input.m[leaf_idx]
                        ρb = input.ρ[leaf_idx]
                        Ab = input.quant[column_idx][leaf_idx]

                        ∇AxfW, ∇AyfW, ∇AzfW, ∇AxbW, ∇AybW, ∇AzbW = _gradient_quantity_accumulation(reference_point, rb, mb, ρb, Ab, hb, K)
                        ∇Axf += ∇AxfW
                        ∇Ayf += ∇AyfW
                        ∇Azf += ∇AzfW
                        ∇Axb += ∇AxbW
                        ∇Ayb += ∇AybW
                        ∇Azb += ∇AzbW
                        A += _quantity_interpolate_accumulation(reference_point, rb, mb, ρb, Ab, hb, K)
                        mWlρ += _ShepardNormalization_accumulation(reference_point, rb, mb, ρb, hb, K)
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
    if iszero(mWlρ)
        return (T(NaN), T(NaN), T(NaN))
    end

    # Shepard normalization
    A /= mWlρ

    # Construct gradient
    ∇Axb *= A
    ∇Ayb *= A
    ∇Azb *= A

    # Final result
    ∇Ax = (∇Axf - ∇Axb)
    ∇Ay = (∇Ayf - ∇Ayb)
    ∇Az = (∇Azf - ∇Azb)
    return (∇Ax, ∇Ay, ∇Az)
end

@inline function _gradient_quantity_interpolate_kernel(input::ITPINPUT, reference_point::NTuple{3, T}, ha :: T, LBVH :: LinearBVH, column_idx :: Int, :: Type{itpSymmetric}) where {ITPINPUT <: AbstractInterpolationInput, T <: AbstractFloat}
    # Prepare for interpolation
    K = input.smoothed_kernel
    Ktyp = typeof(K)
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

    # LBVH data
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

    # Handle empty tree
    if iszero(root)
        nleaf = length(leaf_min[1])
        @inbounds for leaf_idx in 1:nleaf
            hb = input.h[leaf_idx]
            radius = Kvalid * max(ha, hb)
            radius2 = radius * radius
            d2 = NeighborSearch._dist2_to_leaf_aabb(leaf_min, leaf_max, reference_point, leaf_idx)
            if d2 <= radius2
                ########### Found a neighbor, do accumulation ###########
                @inbounds begin
                    rb = (input.x[leaf_idx], input.y[leaf_idx], input.z[leaf_idx])
                    mb = input.m[leaf_idx]
                    ρb = input.ρ[leaf_idx]
                    Ab = input.quant[column_idx][leaf_idx]

                    ∇AxfW, ∇AyfW, ∇AzfW, ∇AxbW, ∇AybW, ∇AzbW = _gradient_quantity_accumulation(reference_point, rb, mb, ρb, Ab, ha, hb, K)
                    ∇Axf += ∇AxfW
                    ∇Ayf += ∇AyfW
                    ∇Azf += ∇AzfW
                    ∇Axb += ∇AxbW
                    ∇Ayb += ∇AybW
                    ∇Azb += ∇AzbW
                    A += _quantity_interpolate_accumulation(reference_point, rb, mb, ρb, Ab, ha, hb, K)
                    mWlρ += _ShepardNormalization_accumulation(reference_point, rb, mb, ρb, ha, hb, K)
                end
                #########################################################
            end
        end
        if iszero(mWlρ)
            return (T(NaN), T(NaN), T(NaN))
        end

        # Shepard normalization
        A /= mWlρ

        # Construct gradient
        ∇Axb *= A
        ∇Ayb *= A
        ∇Azb *= A

        # Final result
        ∇Ax = (∇Axf - ∇Axb)
        ∇Ay = (∇Ayf - ∇Ayb)
        ∇Az = (∇Azf - ∇Azb)
        return (∇Ax, ∇Ay, ∇Az)
    end

    # Start traversal
    node = root
    while node != 0
        rnode = Kvalid * max(ha, node_hmax[node])
        r2node = rnode * rnode
        dist2_node = NeighborSearch._dist2_to_node_aabb(node_min, node_max, reference_point, node)
        if dist2_node <= r2node
            if LL[node]
                @inbounds leaf_idx = L[node]
                hb = input.h[leaf_idx]
                radius = Kvalid * max(ha, hb)
                radius2 = radius * radius
                d2 = NeighborSearch._dist2_to_leaf_aabb(leaf_min, leaf_max, reference_point, leaf_idx)
                if d2 <= radius2
                    ########### Found a neighbor, do accumulation ###########
                    @inbounds begin
                        rb = (input.x[leaf_idx], input.y[leaf_idx], input.z[leaf_idx])
                        mb = input.m[leaf_idx]
                        ρb = input.ρ[leaf_idx]
                        Ab = input.quant[column_idx][leaf_idx]

                        ∇AxfW, ∇AyfW, ∇AzfW, ∇AxbW, ∇AybW, ∇AzbW = _gradient_quantity_accumulation(reference_point, rb, mb, ρb, Ab, ha, hb, K)
                        ∇Axf += ∇AxfW
                        ∇Ayf += ∇AyfW
                        ∇Azf += ∇AzfW
                        ∇Axb += ∇AxbW
                        ∇Ayb += ∇AybW
                        ∇Azb += ∇AzbW
                        A += _quantity_interpolate_accumulation(reference_point, rb, mb, ρb, Ab, ha, hb, K)
                        mWlρ += _ShepardNormalization_accumulation(reference_point, rb, mb, ρb, ha, hb, K)
                    end
                    #########################################################
                end
            end
            if RR[node]
                @inbounds leaf_idx = R[node]
                hb = input.h[leaf_idx]
                radius = Kvalid * max(ha, hb)
                radius2 = radius * radius
                d2 = NeighborSearch._dist2_to_leaf_aabb(leaf_min, leaf_max, reference_point, leaf_idx)
                if d2 <= radius2
                    ########### Found a neighbor, do accumulation ###########
                    @inbounds begin
                        rb = (input.x[leaf_idx], input.y[leaf_idx], input.z[leaf_idx])
                        mb = input.m[leaf_idx]
                        ρb = input.ρ[leaf_idx]
                        Ab = input.quant[column_idx][leaf_idx]

                        ∇AxfW, ∇AyfW, ∇AzfW, ∇AxbW, ∇AybW, ∇AzbW = _gradient_quantity_accumulation(reference_point, rb, mb, ρb, Ab, ha, hb, K)
                        ∇Axf += ∇AxfW
                        ∇Ayf += ∇AyfW
                        ∇Azf += ∇AzfW
                        ∇Axb += ∇AxbW
                        ∇Ayb += ∇AybW
                        ∇Azb += ∇AzbW
                        A += _quantity_interpolate_accumulation(reference_point, rb, mb, ρb, Ab, ha, hb, K)
                        mWlρ += _ShepardNormalization_accumulation(reference_point, rb, mb, ρb, ha, hb, K)
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
    if iszero(mWlρ)
        return (T(NaN), T(NaN), T(NaN))
    end

    # Shepard normalization
    A /= mWlρ

    # Construct gradient
    ∇Axb *= A
    ∇Ayb *= A
    ∇Azb *= A

    # Final result
    ∇Ax = (∇Axf - ∇Axb)
    ∇Ay = (∇Ayf - ∇Ayb)
    ∇Az = (∇Azf - ∇Azb)
    return (∇Ax, ∇Ay, ∇Az)
end
