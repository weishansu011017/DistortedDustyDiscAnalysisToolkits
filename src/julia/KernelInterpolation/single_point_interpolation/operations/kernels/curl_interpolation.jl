@inline function _curl_quantity_interpolate_kernel(input::ITPINPUT, reference_point::NTuple{3, T}, ha :: T, LBVH :: LinearBVH, Ax_column_idx :: Int, Ay_column_idx :: Int, Az_column_idx :: Int, :: Type{itpGather}) where {ITPINPUT <: AbstractInterpolationInput, T <: AbstractFloat}
    # Prepare for interpolation
    K = input.smoothed_kernel
    Ktyp = typeof(K)
    Kvalid = KernelFunctionValid(Ktyp, T)

    # Initialize counter
    ∇Axf :: T = zero(T)
    ∇Ayf :: T = zero(T)
    ∇Azf :: T = zero(T)

    mlρ∂xW :: T = zero(T)
    mlρ∂yW :: T = zero(T)
    mlρ∂zW :: T = zero(T)

    ∇Axb :: T = zero(T)
    ∇Ayb :: T = zero(T)
    ∇Azb :: T = zero(T)

    Ax :: T = zero(T)
    Ay :: T = zero(T)
    Az :: T = zero(T)
    S1 :: T = zero(T)
    S2 :: T = zero(T)

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
                    Axb = input.quant[Ax_column_idx][leaf_idx]
                    Ayb = input.quant[Ay_column_idx][leaf_idx]
                    Azb = input.quant[Az_column_idx][leaf_idx]
                    ∇AxfW, ∇AyfW, ∇AzfW, mblρb∂xW, mblρb∂yW, mblρb∂zW = _curl_quantity_accumulation(reference_point, rb, mb, ρb, Axb, Ayb, Azb, ha, K)
                    Ax += _quantity_interpolate_accumulation(reference_point, rb, mb, ρb, Axb, ha, K)
                    Ay += _quantity_interpolate_accumulation(reference_point, rb, mb, ρb, Ayb, ha, K)
                    Az += _quantity_interpolate_accumulation(reference_point, rb, mb, ρb, Azb, ha, K)

                    ∇Axf += ∇AxfW
                    ∇Ayf += ∇AyfW
                    ∇Azf += ∇AzfW
                    mlρ∂xW += mblρb∂xW
                    mlρ∂yW += mblρb∂yW
                    mlρ∂zW += mblρb∂zW
                    S1b = _ShepardNormalization_accumulation(reference_point, rb, mb, ρb, ha, K)
                    S1 += S1b
                    S2 += S1b * S1b
                end
                #########################################################
            end
        end
        if iszero(S1)
            return (T(NaN), T(NaN), T(NaN))
        end

        # Shepard normalization
        Ax /= S1
        Ay /= S1
        Az /= S1

        # Construct curl
        ∇Axb = Ay * mlρ∂zW - Az * mlρ∂yW
        ∇Ayb = Az * mlρ∂xW - Ax * mlρ∂zW
        ∇Azb = Ax * mlρ∂yW - Ay * mlρ∂xW

        # Final result
        ∇Ax = -(∇Axf - ∇Axb)
        ∇Ay = -(∇Ayf - ∇Ayb)
        ∇Az = -(∇Azf - ∇Azb)

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
                        Axb = input.quant[Ax_column_idx][leaf_idx]
                        Ayb = input.quant[Ay_column_idx][leaf_idx]
                        Azb = input.quant[Az_column_idx][leaf_idx]
                        ∇AxfW, ∇AyfW, ∇AzfW, mblρb∂xW, mblρb∂yW, mblρb∂zW = _curl_quantity_accumulation(reference_point, rb, mb, ρb, Axb, Ayb, Azb, ha, K)
                        Ax += _quantity_interpolate_accumulation(reference_point, rb, mb, ρb, Axb, ha, K)
                        Ay += _quantity_interpolate_accumulation(reference_point, rb, mb, ρb, Ayb, ha, K)
                        Az += _quantity_interpolate_accumulation(reference_point, rb, mb, ρb, Azb, ha, K)

                        ∇Axf += ∇AxfW
                        ∇Ayf += ∇AyfW
                        ∇Azf += ∇AzfW
                        mlρ∂xW += mblρb∂xW
                        mlρ∂yW += mblρb∂yW
                        mlρ∂zW += mblρb∂zW
                        S1b = _ShepardNormalization_accumulation(reference_point, rb, mb, ρb, ha, K)
                        S1 += S1b
                        S2 += S1b * S1b
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
                        Axb = input.quant[Ax_column_idx][leaf_idx]
                        Ayb = input.quant[Ay_column_idx][leaf_idx]
                        Azb = input.quant[Az_column_idx][leaf_idx]
                        ∇AxfW, ∇AyfW, ∇AzfW, mblρb∂xW, mblρb∂yW, mblρb∂zW = _curl_quantity_accumulation(reference_point, rb, mb, ρb, Axb, Ayb, Azb, ha, K)
                        Ax += _quantity_interpolate_accumulation(reference_point, rb, mb, ρb, Axb, ha, K)
                        Ay += _quantity_interpolate_accumulation(reference_point, rb, mb, ρb, Ayb, ha, K)
                        Az += _quantity_interpolate_accumulation(reference_point, rb, mb, ρb, Azb, ha, K)

                        ∇Axf += ∇AxfW
                        ∇Ayf += ∇AyfW
                        ∇Azf += ∇AzfW
                        mlρ∂xW += mblρb∂xW
                        mlρ∂yW += mblρb∂yW
                        mlρ∂zW += mblρb∂zW
                        S1b = _ShepardNormalization_accumulation(reference_point, rb, mb, ρb, ha, K)
                        S1 += S1b
                        S2 += S1b * S1b
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
    if iszero(S1)
        return (T(NaN), T(NaN), T(NaN))
    end

    # Shepard normalization
    Ax /= S1
    Ay /= S1
    Az /= S1

    # Construct curl
    ∇Axb = Ay * mlρ∂zW - Az * mlρ∂yW
    ∇Ayb = Az * mlρ∂xW - Ax * mlρ∂zW
    ∇Azb = Ax * mlρ∂yW - Ay * mlρ∂xW

    # Final result
    ∇Ax = -(∇Axf - ∇Axb)
    ∇Ay = -(∇Ayf - ∇Ayb)
    ∇Az = -(∇Azf - ∇Azb)

    return (∇Ax, ∇Ay, ∇Az)
end

@inline function _curl_quantity_interpolate_kernel(input::ITPINPUT, reference_point::NTuple{3, T}, ha :: T, LBVH :: LinearBVH, Ax_column_idx :: Int, Ay_column_idx :: Int, Az_column_idx :: Int, :: Type{itpScatter}) where {ITPINPUT <: AbstractInterpolationInput, T <: AbstractFloat}
    # Prepare for interpolation
    K = input.smoothed_kernel
    Ktyp = typeof(K)
    Kvalid = KernelFunctionValid(Ktyp, T)

    # Initialize counter
    ∇Axf :: T = zero(T)
    ∇Ayf :: T = zero(T)
    ∇Azf :: T = zero(T)

    mlρ∂xW :: T = zero(T)
    mlρ∂yW :: T = zero(T)
    mlρ∂zW :: T = zero(T)

    ∇Axb :: T = zero(T)
    ∇Ayb :: T = zero(T)
    ∇Azb :: T = zero(T)

    Ax :: T = zero(T)
    Ay :: T = zero(T)
    Az :: T = zero(T)
    S1 :: T = zero(T)
    S2 :: T = zero(T)

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

    # Do traversal
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
                    Axb = input.quant[Ax_column_idx][leaf_idx]
                    Ayb = input.quant[Ay_column_idx][leaf_idx]
                    Azb = input.quant[Az_column_idx][leaf_idx]
                    ∇AxfW, ∇AyfW, ∇AzfW, mblρb∂xW, mblρb∂yW, mblρb∂zW = _curl_quantity_accumulation(reference_point, rb, mb, ρb, Axb, Ayb, Azb, hb, K)
                    Ax += _quantity_interpolate_accumulation(reference_point, rb, mb, ρb, Axb, hb, K)
                    Ay += _quantity_interpolate_accumulation(reference_point, rb, mb, ρb, Ayb, hb, K)
                    Az += _quantity_interpolate_accumulation(reference_point, rb, mb, ρb, Azb, hb, K)

                    ∇Axf += ∇AxfW
                    ∇Ayf += ∇AyfW
                    ∇Azf += ∇AzfW
                    mlρ∂xW += mblρb∂xW
                    mlρ∂yW += mblρb∂yW
                    mlρ∂zW += mblρb∂zW
                    S1b = _ShepardNormalization_accumulation(reference_point, rb, mb, ρb, hb, K)
                    S1 += S1b
                    S2 += S1b * S1b
                end
                #########################################################
            end
        end
        if iszero(S1)
            return (T(NaN), T(NaN), T(NaN))
        end

        # Shepard normalization
        Ax /= S1
        Ay /= S1
        Az /= S1

        # Construct curl
        ∇Axb = Ay * mlρ∂zW - Az * mlρ∂yW
        ∇Ayb = Az * mlρ∂xW - Ax * mlρ∂zW
        ∇Azb = Ax * mlρ∂yW - Ay * mlρ∂xW

        # Final result
        ∇Ax = -(∇Axf - ∇Axb)
        ∇Ay = -(∇Ayf - ∇Ayb)
        ∇Az = -(∇Azf - ∇Azb)
        return (∇Ax, ∇Ay, ∇Az)
    end

    # Start traversal
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
                    ########### Found a neighbor, do accumulation ###########
                    @inbounds begin
                        rb = (input.x[leaf_idx], input.y[leaf_idx], input.z[leaf_idx])
                        mb = input.m[leaf_idx]
                        ρb = input.ρ[leaf_idx]
                        Axb = input.quant[Ax_column_idx][leaf_idx]
                        Ayb = input.quant[Ay_column_idx][leaf_idx]
                        Azb = input.quant[Az_column_idx][leaf_idx]
                        ∇AxfW, ∇AyfW, ∇AzfW, mblρb∂xW, mblρb∂yW, mblρb∂zW = _curl_quantity_accumulation(reference_point, rb, mb, ρb, Axb, Ayb, Azb, hb, K)
                        Ax += _quantity_interpolate_accumulation(reference_point, rb, mb, ρb, Axb, hb, K)
                        Ay += _quantity_interpolate_accumulation(reference_point, rb, mb, ρb, Ayb, hb, K)
                        Az += _quantity_interpolate_accumulation(reference_point, rb, mb, ρb, Azb, hb, K)

                        ∇Axf += ∇AxfW
                        ∇Ayf += ∇AyfW
                        ∇Azf += ∇AzfW
                        mlρ∂xW += mblρb∂xW
                        mlρ∂yW += mblρb∂yW
                        mlρ∂zW += mblρb∂zW
                        S1b = _ShepardNormalization_accumulation(reference_point, rb, mb, ρb, hb, K)
                        S1 += S1b
                        S2 += S1b * S1b
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
                        Axb = input.quant[Ax_column_idx][leaf_idx]
                        Ayb = input.quant[Ay_column_idx][leaf_idx]
                        Azb = input.quant[Az_column_idx][leaf_idx]
                        ∇AxfW, ∇AyfW, ∇AzfW, mblρb∂xW, mblρb∂yW, mblρb∂zW = _curl_quantity_accumulation(reference_point, rb, mb, ρb, Axb, Ayb, Azb, hb, K)
                        Ax += _quantity_interpolate_accumulation(reference_point, rb, mb, ρb, Axb, hb, K)
                        Ay += _quantity_interpolate_accumulation(reference_point, rb, mb, ρb, Ayb, hb, K)
                        Az += _quantity_interpolate_accumulation(reference_point, rb, mb, ρb, Azb, hb, K)

                        ∇Axf += ∇AxfW
                        ∇Ayf += ∇AyfW
                        ∇Azf += ∇AzfW
                        mlρ∂xW += mblρb∂xW
                        mlρ∂yW += mblρb∂yW
                        mlρ∂zW += mblρb∂zW
                        S1b = _ShepardNormalization_accumulation(reference_point, rb, mb, ρb, hb, K)
                        S1 += S1b
                        S2 += S1b * S1b
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
    if iszero(S1)
        return (T(NaN), T(NaN), T(NaN))
    end

    # Shepard normalization
    Ax /= S1
    Ay /= S1
    Az /= S1

    # Construct curl
    ∇Axb = Ay * mlρ∂zW - Az * mlρ∂yW
    ∇Ayb = Az * mlρ∂xW - Ax * mlρ∂zW
    ∇Azb = Ax * mlρ∂yW - Ay * mlρ∂xW

    # Final result
    ∇Ax = -(∇Axf - ∇Axb)
    ∇Ay = -(∇Ayf - ∇Ayb)
    ∇Az = -(∇Azf - ∇Azb)
    return (∇Ax, ∇Ay, ∇Az)
end

@inline function _curl_quantity_interpolate_kernel(input::ITPINPUT, reference_point::NTuple{3, T}, ha :: T, LBVH :: LinearBVH, Ax_column_idx :: Int, Ay_column_idx :: Int, Az_column_idx :: Int, :: Type{itpSymmetric}) where {ITPINPUT <: AbstractInterpolationInput, T <: AbstractFloat}
    # Prepare for interpolation
    K = input.smoothed_kernel
    Ktyp = typeof(K)
    Kvalid = KernelFunctionValid(Ktyp, T)

    # Initialize counter
    ∇Axf :: T = zero(T)
    ∇Ayf :: T = zero(T)
    ∇Azf :: T = zero(T)

    mlρ∂xW :: T = zero(T)
    mlρ∂yW :: T = zero(T)
    mlρ∂zW :: T = zero(T)

    ∇Axb :: T = zero(T)
    ∇Ayb :: T = zero(T)
    ∇Azb :: T = zero(T)

    Ax :: T = zero(T)
    Ay :: T = zero(T)
    Az :: T = zero(T)
    S1 :: T = zero(T)
    S2 :: T = zero(T)

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

    # Do traversal
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
                    Axb = input.quant[Ax_column_idx][leaf_idx]
                    Ayb = input.quant[Ay_column_idx][leaf_idx]
                    Azb = input.quant[Az_column_idx][leaf_idx]
                    ∇AxfW, ∇AyfW, ∇AzfW, mblρb∂xW, mblρb∂yW, mblρb∂zW = _curl_quantity_accumulation(reference_point, rb, mb, ρb, Axb, Ayb, Azb, ha, hb, K)
                    Ax += _quantity_interpolate_accumulation(reference_point, rb, mb, ρb, Axb, ha, hb, K)
                    Ay += _quantity_interpolate_accumulation(reference_point, rb, mb, ρb, Ayb, ha, hb, K)
                    Az += _quantity_interpolate_accumulation(reference_point, rb, mb, ρb, Azb, ha, hb, K)

                    ∇Axf += ∇AxfW
                    ∇Ayf += ∇AyfW
                    ∇Azf += ∇AzfW
                    mlρ∂xW += mblρb∂xW
                    mlρ∂yW += mblρb∂yW
                    mlρ∂zW += mblρb∂zW
                    S1b = _ShepardNormalization_accumulation(reference_point, rb, mb, ρb, ha, hb, K)
                    S1 += S1b
                    S2 += S1b * S1b
                end
                #########################################################
            end
        end
        if iszero(S1)
            return (T(NaN), T(NaN), T(NaN))
        end

        # Shepard normalization
        Ax /= S1
        Ay /= S1
        Az /= S1

        # Construct curl
        ∇Axb = Ay * mlρ∂zW - Az * mlρ∂yW
        ∇Ayb = Az * mlρ∂xW - Ax * mlρ∂zW
        ∇Azb = Ax * mlρ∂yW - Ay * mlρ∂xW

        # Final result
        ∇Ax = -(∇Axf - ∇Axb)
        ∇Ay = -(∇Ayf - ∇Ayb)
        ∇Az = -(∇Azf - ∇Azb)
        return (∇Ax, ∇Ay, ∇Az)
    end

    # Start traversal
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
                    ########### Found a neighbor, do accumulation ###########
                    @inbounds begin
                        rb = (input.x[leaf_idx], input.y[leaf_idx], input.z[leaf_idx])
                        mb = input.m[leaf_idx]
                        ρb = input.ρ[leaf_idx]
                        Axb = input.quant[Ax_column_idx][leaf_idx]
                        Ayb = input.quant[Ay_column_idx][leaf_idx]
                        Azb = input.quant[Az_column_idx][leaf_idx]
                        ∇AxfW, ∇AyfW, ∇AzfW, mblρb∂xW, mblρb∂yW, mblρb∂zW = _curl_quantity_accumulation(reference_point, rb, mb, ρb, Axb, Ayb, Azb, ha, hb, K)
                        Ax += _quantity_interpolate_accumulation(reference_point, rb, mb, ρb, Axb, ha, hb, K)
                        Ay += _quantity_interpolate_accumulation(reference_point, rb, mb, ρb, Ayb, ha, hb, K)
                        Az += _quantity_interpolate_accumulation(reference_point, rb, mb, ρb, Azb, ha, hb, K)

                        ∇Axf += ∇AxfW
                        ∇Ayf += ∇AyfW
                        ∇Azf += ∇AzfW
                        mlρ∂xW += mblρb∂xW
                        mlρ∂yW += mblρb∂yW
                        mlρ∂zW += mblρb∂zW
                        S1b = _ShepardNormalization_accumulation(reference_point, rb, mb, ρb, ha, hb, K)
                        S1 += S1b
                        S2 += S1b * S1b
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
                        Axb = input.quant[Ax_column_idx][leaf_idx]
                        Ayb = input.quant[Ay_column_idx][leaf_idx]
                        Azb = input.quant[Az_column_idx][leaf_idx]
                        ∇AxfW, ∇AyfW, ∇AzfW, mblρb∂xW, mblρb∂yW, mblρb∂zW = _curl_quantity_accumulation(reference_point, rb, mb, ρb, Axb, Ayb, Azb, ha, hb, K)
                        Ax += _quantity_interpolate_accumulation(reference_point, rb, mb, ρb, Axb, ha, hb, K)
                        Ay += _quantity_interpolate_accumulation(reference_point, rb, mb, ρb, Ayb, ha, hb, K)
                        Az += _quantity_interpolate_accumulation(reference_point, rb, mb, ρb, Azb, ha, hb, K)

                        ∇Axf += ∇AxfW
                        ∇Ayf += ∇AyfW
                        ∇Azf += ∇AzfW
                        mlρ∂xW += mblρb∂xW
                        mlρ∂yW += mblρb∂yW
                        mlρ∂zW += mblρb∂zW
                        S1b = _ShepardNormalization_accumulation(reference_point, rb, mb, ρb, ha, hb, K)
                        S1 += S1b
                        S2 += S1b * S1b
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
    if iszero(S1)
        return (T(NaN), T(NaN), T(NaN))
    end

    # Shepard normalization
    Ax /= S1
    Ay /= S1
    Az /= S1

    # Construct curl
    ∇Axb = Ay * mlρ∂zW - Az * mlρ∂yW
    ∇Ayb = Az * mlρ∂xW - Ax * mlρ∂zW
    ∇Azb = Ax * mlρ∂yW - Ay * mlρ∂xW

    # Final result
    ∇Ax = -(∇Axf - ∇Axb)
    ∇Ay = -(∇Ayf - ∇Ayb)
    ∇Az = -(∇Azf - ∇Azb)
    return (∇Ax, ∇Ay, ∇Az)
end