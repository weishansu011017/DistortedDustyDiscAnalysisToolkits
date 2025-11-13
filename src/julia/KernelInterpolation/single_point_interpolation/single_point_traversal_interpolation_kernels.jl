# Determine interpolation type
@enum InterpolationStrategy begin
    itpGather
    itpScatter
    itpSymmetric
end

## Density
@inline function _density_accumulation(input :: ITPINPUT, reference_point::NTuple{3, T}, ha :: T, itp_strategy :: InterpolationStrategy, i :: Int) where {ITPINPUT <: AbstractInterpolationInput, T <: AbstractFloat}
    rb :: NTuple{3, T} = (input.x[i], input.y[i], input.z[i])
    Ktyp = typeof(input.smoothed_kernel)
    W :: T = zero(T)
    
    if itp_strategy == itpGather
        W = Smoothed_kernel_function(Ktyp, reference_point, rb, ha)
    elseif itp_strategy == itpScatter
        W = Smoothed_kernel_function(Ktyp, reference_point, rb, input.h[i])
    elseif itp_strategy == itpSymmetric
        W = T(0.5) * (Smoothed_kernel_function(Ktyp, reference_point, rb, ha) + Smoothed_kernel_function(Ktyp, reference_point, rb, input.h[i]))
    end
    return input.m[i] * W
end
@inline function _density_kernel(input::ITPINPUT, reference_point::NTuple{3, T}, ha :: T, LBVH :: LinearBVH, itp_strategy :: InterpolationStrategy = itpSymmetric) where {ITPINPUT <: AbstractInterpolationInput, T <: AbstractFloat}
    # Prepare for interpolation
    Ktyp = typeof(input.smoothed_kernel)
    Kvalid = KernelFunctionValid(Ktyp, T)

    # Initialize counter
    rho :: T = zero(T)

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
    if root == 0
        nleaf = length(leaf_min[1])
        @inbounds for leaf_idx in 1:nleaf
            d2 = _dist2_to_leaf_aabb(leaf_min, leaf_max, reference_point, leaf_idx)
            if d2 <= radius2
                ########### Found a neighbor, do accumulation ###########
                rho += _density_accumulation(input, reference_point, ha, itp_strategy, leaf_idx)
                #########################################################
            end
        end
        return rho
    end

    # Start traversal
    node = root
    while node != 0
        dist2_node = _dist2_to_node_aabb(node_min, node_max, reference_point, node)
        if dist2_node <= radius2
            if LL[node]
                @inbounds leaf_idx = L[node]
                d2 = _dist2_to_leaf_aabb(leaf_min, leaf_max, reference_point, leaf_idx)
                if d2 <= radius2
                    ########### Found a neighbor, do accumulation ###########
                    rho += _density_accumulation(input, reference_point, ha, itp_strategy, leaf_idx)
                    #########################################################
                end
            end
            if RR[node]
                @inbounds leaf_idx = R[node]
                d2 = _dist2_to_leaf_aabb(leaf_min, leaf_max, reference_point, leaf_idx)
                if d2 <= radius2
                    ########### Found a neighbor, do accumulation ###########
                    rho += _density_accumulation(input, reference_point, ha, itp_strategy, leaf_idx)
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

            node = _next_internal_node(node, L, R, LL, RR, node_parent)
        else
            node = _next_internal_node(node, L, R, LL, RR, node_parent)
        end
    end
    return rho
end

## Number density
@inline function _number_density_accumulation(input :: ITPINPUT, reference_point::NTuple{3, T}, ha :: T, itp_strategy :: InterpolationStrategy, i :: Int) where {ITPINPUT <: AbstractInterpolationInput, T <: AbstractFloat}
    rb :: NTuple{3, T} = (input.x[i], input.y[i], input.z[i])
    Ktyp = typeof(input.smoothed_kernel)
    W :: T = zero(T)
    
    if itp_strategy == itpGather
        W = Smoothed_kernel_function(Ktyp, reference_point, rb, ha)
    elseif itp_strategy == itpScatter
        W = Smoothed_kernel_function(Ktyp, reference_point, rb, input.h[i])
    elseif itp_strategy == itpSymmetric
        W = T(0.5) * (Smoothed_kernel_function(Ktyp, reference_point, rb, ha) + Smoothed_kernel_function(Ktyp, reference_point, rb, input.h[i]))
    end
    return W
end
@inline function _number_density_kernel(input::ITPINPUT, reference_point::NTuple{3, T}, ha :: T, LBVH :: LinearBVH, itp_strategy :: InterpolationStrategy = itpSymmetric) where {ITPINPUT <: AbstractInterpolationInput, T <: AbstractFloat}
    # Prepare for interpolation
    Ktyp = typeof(input.smoothed_kernel)
    Kvalid = KernelFunctionValid(Ktyp, T)

    # Initialize counter
    n :: T = zero(T)

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
    if root == 0
        nleaf = length(leaf_min[1])
        @inbounds for leaf_idx in 1:nleaf
            d2 = _dist2_to_leaf_aabb(leaf_min, leaf_max, reference_point, leaf_idx)
            if d2 <= radius2
                ########### Found a neighbor, do accumulation ###########
                n += _number_density_accumulation(input, reference_point, ha, itp_strategy, leaf_idx)
                #########################################################
            end
        end
        return n
    end

    # Start traversal
    node = root
    while node != 0
        dist2_node = _dist2_to_node_aabb(node_min, node_max, reference_point, node)
        if dist2_node <= radius2
            if LL[node]
                @inbounds leaf_idx = L[node]
                d2 = _dist2_to_leaf_aabb(leaf_min, leaf_max, reference_point, leaf_idx)
                if d2 <= radius2
                    ########### Found a neighbor, do accumulation ###########
                    n += _number_density_accumulation(input, reference_point, ha, itp_strategy, leaf_idx)
                    #########################################################
                end
            end
            if RR[node]
                @inbounds leaf_idx = R[node]
                d2 = _dist2_to_leaf_aabb(leaf_min, leaf_max, reference_point, leaf_idx)
                if d2 <= radius2
                    ########### Found a neighbor, do accumulation ###########
                    n += _number_density_accumulation(input, reference_point, ha, itp_strategy, leaf_idx)
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

            node = _next_internal_node(node, L, R, LL, RR, node_parent)
        else
            node = _next_internal_node(node, L, R, LL, RR, node_parent)
        end
    end
    return n
end

## Single quantity intepolation
@inline function _quantity_interpolate_accumulation(input :: ITPINPUT, reference_point::NTuple{3, T}, ha :: T, column_idx :: Int64, itp_strategy :: InterpolationStrategy, i :: Int) where {ITPINPUT <: AbstractInterpolationInput, T <: AbstractFloat}
    Ktyp = typeof(input.smoothed_kernel)
    mb = input.m[i]
    ρb = input.ρ[i]
    Ab = input.quant[column_idx][i]
    rb :: NTuple{3, T} = (input.x[i], input.y[i], input.z[i])
    W :: T = zero(T)
    if itp_strategy == itpGather
        W = Smoothed_kernel_function(Ktyp, reference_point, rb, ha)
    elseif itp_strategy == itpScatter
        W = Smoothed_kernel_function(Ktyp, reference_point, rb, input.h[i])
    elseif itp_strategy == itpSymmetric
        W = T(0.5) * (Smoothed_kernel_function(Ktyp, reference_point, rb, ha) + Smoothed_kernel_function(Ktyp, reference_point, rb, input.h[i]))
    end

    mbWlρb = mb * W/ρb
    return Ab * mbWlρb
end
@inline function _ShepardNormalization_accumulation(input :: ITPINPUT, reference_point::NTuple{3, T}, ha :: T, itp_strategy :: InterpolationStrategy, i :: Int) where {ITPINPUT <: AbstractInterpolationInput, T <: AbstractFloat}
    Ktyp = typeof(input.smoothed_kernel)
    mb = input.m[i]
    ρb = input.ρ[i]
    rb :: NTuple{3, T} = (input.x[i], input.y[i], input.z[i])
    W :: T = zero(T)
    if itp_strategy == itpGather
        W = Smoothed_kernel_function(Ktyp, reference_point, rb, ha)
    elseif itp_strategy == itpScatter
        W = Smoothed_kernel_function(Ktyp, reference_point, rb, input.h[i])
    elseif itp_strategy == itpSymmetric
        W = T(0.5) * (Smoothed_kernel_function(Ktyp, reference_point, rb, ha) + Smoothed_kernel_function(Ktyp, reference_point, rb, input.h[i]))
    end

    mbWlρb = mb * W/ρb
    return mbWlρb
end

@inline function _quantity_interpolate_kernel(input::ITPINPUT, reference_point::NTuple{3, T}, ha :: T, LBVH :: LinearBVH, column_idx :: Int64, ShepardNormalization :: Bool, itp_strategy :: InterpolationStrategy = itpSymmetric) where {ITPINPUT <: AbstractInterpolationInput, T <: AbstractFloat}
    # Prepare for interpolation
    Ktyp = typeof(input.smoothed_kernel)
    Kvalid = KernelFunctionValid(Ktyp, T)

    # Initialize counter
    A :: T = zero(T)
    mWlρ :: T = zero(T)

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
    if root == 0
        nleaf = length(leaf_min[1])
        @inbounds for leaf_idx in 1:nleaf
            d2 = _dist2_to_leaf_aabb(leaf_min, leaf_max, reference_point, leaf_idx)
            if d2 <= radius2
                ########### Found a neighbor, do accumulation ###########
                A += _quantity_interpolate_accumulation(input, reference_point, ha, column_idx, itp_strategy, leaf_idx)
                mWlρ += _ShepardNormalization_accumulation(input, reference_point, ha, itp_strategy, leaf_idx)
                #########################################################
            end
        end
        # Shepard normalization
        if ShepardNormalization
            A /= mWlρ
        end
        return A
    end

    # Start traversal
    node = root
    while node != 0
        dist2_node = _dist2_to_node_aabb(node_min, node_max, reference_point, node)
        if dist2_node <= radius2
            if LL[node]
                @inbounds leaf_idx = L[node]
                d2 = _dist2_to_leaf_aabb(leaf_min, leaf_max, reference_point, leaf_idx)
                if d2 <= radius2
                    ########### Found a neighbor, do accumulation ###########
                    A += _quantity_interpolate_accumulation(input, reference_point, ha, column_idx, itp_strategy, leaf_idx)
                    mWlρ += _ShepardNormalization_accumulation(input, reference_point, ha, itp_strategy, leaf_idx)
                    #########################################################
                end
            end
            if RR[node]
                @inbounds leaf_idx = R[node]
                d2 = _dist2_to_leaf_aabb(leaf_min, leaf_max, reference_point, leaf_idx)
                if d2 <= radius2
                    ########### Found a neighbor, do accumulation ###########
                    A += _quantity_interpolate_accumulation(input, reference_point, ha, column_idx, itp_strategy, leaf_idx)
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

            node = _next_internal_node(node, L, R, LL, RR, node_parent)
        else
            node = _next_internal_node(node, L, R, LL, RR, node_parent)
        end
    end
    # Shepard normalization
    if ShepardNormalization
        A /= mWlρ
    end
    return A
end

## Muti-columns intepolation
@inline function _quantities_interpolate_kernel!(output :: O, input::ITPINPUT, reference_point::NTuple{3, T}, ha :: T, LBVH :: LinearBVH, columns::NTuple{M,Int}, ShepardNormalization :: NTuple{M, Bool}, itp_strategy :: InterpolationStrategy) where {ITPINPUT <: AbstractInterpolationInput, T <: AbstractFloat, O<:AbstractVector{T}, M}
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
    if root == 0
        nleaf = length(leaf_min[1])
        @inbounds for leaf_idx in 1:nleaf
            d2 = _dist2_to_leaf_aabb(leaf_min, leaf_max, reference_point, leaf_idx)
            if d2 <= radius2
                ########### Found a neighbor, do accumulation ###########
                mWlρ += _ShepardNormalization_accumulation(input, reference_point, ha, itp_strategy, leaf_idx)
                @inbounds for j in 1:M
                    column_idx = columns[j]
                    output[j] += _quantity_interpolate_accumulation(input, reference_point, ha, column_idx, itp_strategy, leaf_idx)
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
        return nothing
    end

    # Start traversal
    node = root
    while node != 0
        dist2_node = _dist2_to_node_aabb(node_min, node_max, reference_point, node)
        if dist2_node <= radius2
            if LL[node]
                @inbounds leaf_idx = L[node]
                d2 = _dist2_to_leaf_aabb(leaf_min, leaf_max, reference_point, leaf_idx)
                if d2 <= radius2
                    ########### Found a neighbor, do accumulation ###########
                    mWlρ += _ShepardNormalization_accumulation(input, reference_point, ha, itp_strategy, leaf_idx)
                    @inbounds for j in 1:M
                        column_idx = columns[j]
                        output[j] += _quantity_interpolate_accumulation(input, reference_point, ha, column_idx, itp_strategy, leaf_idx)
                    end
                    #########################################################
                end
            end
            if RR[node]
                @inbounds leaf_idx = R[node]
                d2 = _dist2_to_leaf_aabb(leaf_min, leaf_max, reference_point, leaf_idx)
                if d2 <= radius2
                    ########### Found a neighbor, do accumulation ###########
                    mWlρ += _ShepardNormalization_accumulation(input, reference_point, ha, itp_strategy, leaf_idx)
                    @inbounds for j in 1:M
                        column_idx = columns[j]
                        output[j] += _quantity_interpolate_accumulation(input, reference_point, ha, column_idx, itp_strategy, leaf_idx)
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

            node = _next_internal_node(node, L, R, LL, RR, node_parent)
        else
            node = _next_internal_node(node, L, R, LL, RR, node_parent)
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


@inline function _quantities_interpolate_kernel!(output :: O, input::ITPINPUT, reference_point::NTuple{3, T}, ha :: T, LBVH :: LinearBVH, itp_strategy :: InterpolationStrategy = itpSymmetric) where {ITPINPUT <: AbstractInterpolationInput, T <: AbstractFloat, O<:AbstractVector{T}}
    val_len = Val(length(input.quant))
    columns = ntuple(identity, val_len)
    ShepardNormalization = ntuple(_ -> true, val_len)
    return _quantities_interpolate_kernel!(output, input, reference_point, ha, LBVH, columns, ShepardNormalization, itp_strategy)
end

## LOS density interpolation (Column / Surface density)
@inline function _LOS_density_accumulation(input :: ITPINPUT, reference_point::NTuple{2, T}, ha :: T, itp_strategy :: InterpolationStrategy, i :: Int) where {ITPINPUT <: AbstractInterpolationInput, T <: AbstractFloat}
    rb :: NTuple{2, T} = (input.x[i], input.y[i])
    Ktyp = typeof(input.smoothed_kernel)
    W :: T = zero(T)
    
    if itp_strategy == itpGather
        W = LOSint_Smoothed_kernel_function(Ktyp, reference_point, rb, ha)
    elseif itp_strategy == itpScatter
        W = LOSint_Smoothed_kernel_function(Ktyp, reference_point, rb, input.h[i])
    elseif itp_strategy == itpSymmetric
        W = T(0.5) * (LOSint_Smoothed_kernel_function(Ktyp, reference_point, rb, ha) + LOSint_Smoothed_kernel_function(Ktyp, reference_point, rb, input.h[i]))
    end
    return input.m[i] * W
end
@inline function _LOS_density_kernel(input::ITPINPUT, reference_point::NTuple{2, T}, ha :: T, LBVH :: LinearBVH, itp_strategy :: InterpolationStrategy = itpSymmetric) where {ITPINPUT <: AbstractInterpolationInput, T <: AbstractFloat}
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
    if root == 0
        nleaf = length(leaf_min[1])
        @inbounds for leaf_idx in 1:nleaf
            d2 = _dist2_to_leaf_aabb(leaf_min, leaf_max, reference_point, leaf_idx)
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
        dist2_node = _dist2_to_node_aabb(node_min, node_max, reference_point, node)
        if dist2_node <= radius2
            if LL[node]
                @inbounds leaf_idx = L[node]
                d2 = _dist2_to_leaf_aabb(leaf_min, leaf_max, reference_point, leaf_idx)
                if d2 <= radius2
                    ########### Found a neighbor, do accumulation ###########
                    Sigma += _LOS_density_accumulation(input, reference_point, ha, itp_strategy, leaf_idx)
                    #########################################################
                end
            end
            if RR[node]
                @inbounds leaf_idx = R[node]
                d2 = _dist2_to_leaf_aabb(leaf_min, leaf_max, reference_point, leaf_idx)
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

            node = _next_internal_node(node, L, R, LL, RR, node_parent)
        else
            node = _next_internal_node(node, L, R, LL, RR, node_parent)
        end
    end
    return Sigma
end

## LOS quantities interpolation
@inline function _LOS_quantity_interpolate_accumulation(input :: ITPINPUT, reference_point::NTuple{2, T}, ha :: T, column_idx :: Int64, itp_strategy :: InterpolationStrategy, i :: Int) where {ITPINPUT <: AbstractInterpolationInput, T <: AbstractFloat}
    rb :: NTuple{2, T} = (input.x[i], input.y[i])
    Ktyp = typeof(input.smoothed_kernel)
    W :: T = zero(T)

    mb = input.m[i]
    ρb = input.ρ[i]
    Ab = input.quant[column_idx][i]
    rb :: NTuple{2, T} = (input.x[i], input.y[i])
    W :: T = zero(T)
    if itp_strategy == itpGather
        W = LOSint_Smoothed_kernel_function(Ktyp, reference_point, rb, ha)
    elseif itp_strategy == itpScatter
        W = LOSint_Smoothed_kernel_function(Ktyp, reference_point, rb, input.h[i])
    elseif itp_strategy == itpSymmetric
        W = T(0.5) * (LOSint_Smoothed_kernel_function(Ktyp, reference_point, rb, ha) + LOSint_Smoothed_kernel_function(Ktyp, reference_point, rb, input.h[i]))
    end

    mbWlρb = mb * W/ρb
    return Ab * mbWlρb
end
@inline function _LOS_ShepardNormalization_accumulation(input :: ITPINPUT, reference_point::NTuple{2, T}, ha :: T, itp_strategy :: InterpolationStrategy, i :: Int) where {ITPINPUT <: AbstractInterpolationInput, T <: AbstractFloat}
    rb :: NTuple{2, T} = (input.x[i], input.y[i])
    Ktyp = typeof(input.smoothed_kernel)
    W :: T = zero(T)

    mb = input.m[i]
    ρb = input.ρ[i]
    rb :: NTuple{2, T} = (input.x[i], input.y[i])
    W :: T = zero(T)
    if itp_strategy == itpGather
        W = LOSint_Smoothed_kernel_function(Ktyp, reference_point, rb, ha)
    elseif itp_strategy == itpScatter
        W = LOSint_Smoothed_kernel_function(Ktyp, reference_point, rb, input.h[i])
    elseif itp_strategy == itpSymmetric
        W = T(0.5) * (LOSint_Smoothed_kernel_function(Ktyp, reference_point, rb, ha) + LOSint_Smoothed_kernel_function(Ktyp, reference_point, rb, input.h[i]))
    end

    mbWlρb = mb * W/ρb
    return mbWlρb
end
@inline function _LOS_quantities_interpolate_kernel!(output :: O, input::ITPINPUT, reference_point::NTuple{2, T}, ha :: T, LBVH :: LinearBVH, columns::NTuple{M,Int}, ShepardNormalization :: NTuple{M, Bool}, itp_strategy :: InterpolationStrategy) where {ITPINPUT <: AbstractInterpolationInput, T <: AbstractFloat, O<:AbstractVector{T}, M}
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
    if root == 0
        nleaf = length(leaf_min[1])
        @inbounds for leaf_idx in 1:nleaf
            d2 = _dist2_to_leaf_aabb(leaf_min, leaf_max, reference_point, leaf_idx)
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
        dist2_node = _dist2_to_node_aabb(node_min, node_max, reference_point, node)
        if dist2_node <= radius2
            if LL[node]
                @inbounds leaf_idx = L[node]
                d2 = _dist2_to_leaf_aabb(leaf_min, leaf_max, reference_point, leaf_idx)
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
                d2 = _dist2_to_leaf_aabb(leaf_min, leaf_max, reference_point, leaf_idx)
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

            node = _next_internal_node(node, L, R, LL, RR, node_parent)
        else
            node = _next_internal_node(node, L, R, LL, RR, node_parent)
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

@inline function _LOS_quantities_interpolate_kernel!(output :: O, input::ITPINPUT, reference_point::NTuple{2, T}, ha :: T, LBVH :: LinearBVH, itp_strategy :: InterpolationStrategy = itpSymmetric) where {ITPINPUT <: AbstractInterpolationInput, T <: AbstractFloat, O<:AbstractVector{T}}
    val_len = Val(length(input.quant))
    columns = ntuple(identity, val_len)
    ShepardNormalization = ntuple(_ -> true, val_len)
    return _LOS_quantities_interpolate_kernel!(output, input, reference_point, ha, LBVH, columns, ShepardNormalization, itp_strategy)
end

"""
∇ρ(r) = (1/ρ(r))∑_b m_b*(ρ_b-ρ(r))∇W(r-r_b)
      = (1/ρ(r))((∑_b m_b*ρ_b*∇W(r-r_b))  - ρ(r)(∑_b m_b*∇W(r-r_b))
      = (1/ρ(r))((∑_b m_b*ρ_b*∇W(r-r_b)) - ∑_b m_b*∇W(r-r_b)
"""
# Single column gradient density intepolation
@inline function _gradient_density_accumulation(input :: ITPINPUT, reference_point::NTuple{3, T}, ha :: T, itp_strategy :: InterpolationStrategy, i :: Int) where {ITPINPUT <: AbstractInterpolationInput, T <: AbstractFloat}
    rb :: NTuple{3, T} = (input.x[i], input.y[i], input.z[i])
    Ktyp = typeof(input.smoothed_kernel)
    W :: T = zero(T)
    
    if itp_strategy == itpGather
        W = Smoothed_kernel_function(Ktyp, reference_point, rb, ha)
    elseif itp_strategy == itpScatter
        W = Smoothed_kernel_function(Ktyp, reference_point, rb, input.h[i])
    elseif itp_strategy == itpSymmetric
        W = T(0.5) * (Smoothed_kernel_function(Ktyp, reference_point, rb, ha) + Smoothed_kernel_function(Ktyp, reference_point, rb, input.h[i]))
    end

    mb = input.m[i]
    ρb = input.ρ[i]
    rb :: NTuple{3, T} = (input.x[i], input.y[i], input.z[i])
    W :: T = zero(T)
    if itp_strategy == itpGather
        W = Smoothed_kernel_function(Ktyp, reference_point, rb, ha)
    elseif itp_strategy == itpScatter
        W = Smoothed_kernel_function(Ktyp, reference_point, rb, input.h[i])
    elseif itp_strategy == itpSymmetric
        W = T(0.5) * (Smoothed_kernel_function(Ktyp, reference_point, rb, ha) + Smoothed_kernel_function(Ktyp, reference_point, rb, input.h[i]))
    end
    ∂xW :: T = zero(T)
    ∂yW :: T = zero(T)
    ∂zW :: T = zero(T)
    if itp_strategy == itpGather
        ∇W = Smoothed_gradient_kernel_function(Ktyp, reference_point, rb, ha)
        ∂xW = ∇W[1]
        ∂yW = ∇W[2]
        ∂zW = ∇W[3]
    elseif itp_strategy == itpScatter
        ∇W = Smoothed_gradient_kernel_function(Ktyp, reference_point, rb, input.h[i])
        ∂xW = ∇W[1]
        ∂yW = ∇W[2]
        ∂zW = ∇W[3]
    elseif itp_strategy == itpSymmetric
        ∇Wa = Smoothed_gradient_kernel_function(Ktyp, reference_point, rb, ha)
        ∇Wb = Smoothed_gradient_kernel_function(Ktyp, reference_point, rb, input.h[i])
        ∂xW = T(0.5) * (∇Wa[1] + ∇Wb[1])
        ∂yW = T(0.5) * (∇Wa[2] + ∇Wb[2])
        ∂zW = T(0.5) * (∇Wa[3] + ∇Wb[3])
    end

    # Gradient
    mb∂xW = mb * ∂xW
    mb∂yW = mb * ∂yW
    mb∂zW = mb * ∂zW

    ∇ρxf = mb∂xW * ρb
    ∇ρyf = mb∂yW * ρb
    ∇ρzf = mb∂zW * ρb
    ∇ρxb = mb∂xW
    ∇ρyb = mb∂yW
    ∇ρzb = mb∂zW
    return ∇ρxf, ∇ρyf, ∇ρzf, ∇ρxb, ∇ρyb, ∇ρzb
end
@inline function _gradient_density_kernel(input::ITPINPUT, reference_point::NTuple{3, T}, ha :: T, LBVH :: LinearBVH, itp_strategy :: InterpolationStrategy = itpSymmetric) where {ITPINPUT <: AbstractInterpolationInput, T <: AbstractFloat}
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
    if root == 0
        nleaf = length(leaf_min[1])
        @inbounds for leaf_idx in 1:nleaf
            d2 = _dist2_to_leaf_aabb(leaf_min, leaf_max, reference_point, leaf_idx)
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
        dist2_node = _dist2_to_node_aabb(node_min, node_max, reference_point, node)
        if dist2_node <= radius2
            if LL[node]
                @inbounds leaf_idx = L[node]
                d2 = _dist2_to_leaf_aabb(leaf_min, leaf_max, reference_point, leaf_idx)
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
                d2 = _dist2_to_leaf_aabb(leaf_min, leaf_max, reference_point, leaf_idx)
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

            node = _next_internal_node(node, L, R, LL, RR, node_parent)
        else
            node = _next_internal_node(node, L, R, LL, RR, node_parent)
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

"""
∇A(r) = (1/ρ(r))∑_b m_b*(A_b-A(r))∇W(r-r_b)
      = (1/ρ(r))((∑_b m_b*A_b*∇W(r-r_b))  - A(r)(∑_b m_b*∇W(r-r_b))
      = ∇Af - ∇Ab
"""
# Single column gradient value intepolation
@inline function _gradient_quantity_accumulation(input :: ITPINPUT, reference_point::NTuple{3, T}, ha :: T, column_idx :: Int64, itp_strategy :: InterpolationStrategy, i :: Int) where {ITPINPUT <: AbstractInterpolationInput, T <: AbstractFloat}
    Ktyp = typeof(input.smoothed_kernel)
    mb = input.m[i]
    Ab = input.quant[column_idx][i]
    rb :: NTuple{3, T} = (input.x[i], input.y[i], input.z[i])
    W :: T = zero(T)
    if itp_strategy == itpGather
        W = Smoothed_kernel_function(Ktyp, reference_point, rb, ha)
    elseif itp_strategy == itpScatter
        W = Smoothed_kernel_function(Ktyp, reference_point, rb, input.hs[i])
    elseif itp_strategy == itpSymmetric
        W = T(0.5) * (Smoothed_kernel_function(Ktyp, reference_point, rb, ha) + Smoothed_kernel_function(Ktyp, reference_point, rb, input.hs[i]))
    end
    ∂xW :: T = zero(T)
    ∂yW :: T = zero(T)
    ∂zW :: T = zero(T)
    if itp_strategy == itpGather
        ∇W = Smoothed_gradient_kernel_function(Ktyp, reference_point, rb, ha)
        ∂xW = ∇W[1]
        ∂yW = ∇W[2]
        ∂zW = ∇W[3]
    elseif itp_strategy == itpScatter
        ∇W = Smoothed_gradient_kernel_function(Ktyp, reference_point, rb, input.hs[i])
        ∂xW = ∇W[1]
        ∂yW = ∇W[2]
        ∂zW = ∇W[3]
    elseif itp_strategy == itpSymmetric
        ∇Wa = Smoothed_gradient_kernel_function(Ktyp, reference_point, rb, ha)
        ∇Wb = Smoothed_gradient_kernel_function(Ktyp, reference_point, rb, input.hs[i])
        ∂xW = T(0.5) * (∇Wa[1] + ∇Wb[1])
        ∂yW = T(0.5) * (∇Wa[2] + ∇Wb[2])
        ∂zW = T(0.5) * (∇Wa[3] + ∇Wb[3])
    end

    # Gradient
    mb∂xW = mb * ∂xW
    mb∂yW = mb * ∂yW
    mb∂zW = mb * ∂zW

    ∇Axf = mb∂xW * Ab
    ∇Ayf = mb∂yW * Ab
    ∇Azf = mb∂zW * Ab
    ∇Axb = mb∂xW
    ∇Ayb = mb∂yW
    ∇Azb = mb∂zW
    return ∇Axf, ∇Ayf, ∇Azf, ∇Axb, ∇Ayb, ∇Azb
end
@inline function _gradient_quantity_interpolate_kernel(input::ITPINPUT, reference_point::NTuple{3, T}, ha :: T, LBVH :: LinearBVH, column_idx :: Int64, itp_strategy :: InterpolationStrategy = itpSymmetric) where {ITPINPUT <: AbstractInterpolationInput, T <: AbstractFloat}
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
    if root == 0
        nleaf = length(leaf_min[1])
        @inbounds for leaf_idx in 1:nleaf
            d2 = _dist2_to_leaf_aabb(leaf_min, leaf_max, reference_point, leaf_idx)
            if d2 <= radius2
                ########### Found a neighbor, do accumulation ###########
                ∇AxfW, ∇AyfW, ∇AzfW, ∇AxbW, ∇AybW, ∇AzbW = _gradient_quantity_accumulation(input, reference_point, ha, column_idx, itp_strategy, leaf_idx)
                ρ += _density_accumulation(input, reference_point, ha, itp_strategy, leaf_idx)
                A += _quantity_accumulation(input, reference_point, ha, column_idx, itp_strategy, leaf_idx)
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
        dist2_node = _dist2_to_node_aabb(node_min, node_max, reference_point, node)
        if dist2_node <= radius2
            if LL[node]
                @inbounds leaf_idx = L[node]
                d2 = _dist2_to_leaf_aabb(leaf_min, leaf_max, reference_point, leaf_idx)
                if d2 <= radius2
                    ########### Found a neighbor, do accumulation ###########
                    ∇AxfW, ∇AyfW, ∇AzfW, ∇AxbW, ∇AybW, ∇AzbW = _gradient_quantity_accumulation(input, reference_point, ha, column_idx, itp_strategy, leaf_idx)
                    ρ += _density_accumulation(input, reference_point, ha, itp_strategy, leaf_idx)
                    A += _quantity_accumulation(input, reference_point, ha, column_idx, itp_strategy, leaf_idx)
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
                d2 = _dist2_to_leaf_aabb(leaf_min, leaf_max, reference_point, leaf_idx)
                if d2 <= radius2
                    ########### Found a neighbor, do accumulation ###########
                    ∇AxfW, ∇AyfW, ∇AzfW, ∇AxbW, ∇AybW, ∇AzbW = _gradient_quantity_accumulation(input, reference_point, ha, column_idx, itp_strategy, leaf_idx)
                    ρ += _density_accumulation(input, reference_point, ha, itp_strategy, leaf_idx)
                    A += _quantity_accumulation(input, reference_point, ha, column_idx, itp_strategy, leaf_idx)
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

            node = _next_internal_node(node, L, R, LL, RR, node_parent)
        else
            node = _next_internal_node(node, L, R, LL, RR, node_parent)
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

"""
    ∇⋅A(r) = (1/ρ(r))∑_b m_b*(A_b-A(r))⋅∇W(r-r_b)
           = (1/ρ(r)) * ((∑_b m_b*A_b⋅∇W(r-r_b)))- A(r)⋅(∑_b m_b*∇W(r-r_b)))
           = ∇⋅A(r)
"""
# Single column divergence value intepolation
@inline function _divergence_quantity_accumulation(input :: ITPINPUT, reference_point::NTuple{3, T}, ha :: T, Ax_column_idx :: Int, Ay_column_idx :: Int, Az_column_idx :: Int, itp_strategy :: InterpolationStrategy, i :: Int) where {ITPINPUT <: AbstractInterpolationInput, T <: AbstractFloat}
    Ktyp = typeof(input.smoothed_kernel)
    mb = input.m[i]
    Axb = input.quant[Ax_column_idx][i]
    Ayb = input.quant[Ay_column_idx][i]
    Azb = input.quant[Az_column_idx][i]
    rb :: NTuple{3, T} = (input.x[i], input.y[i], input.z[i])
    W :: T = zero(T)
    if itp_strategy == itpGather
        W = Smoothed_kernel_function(Ktyp, reference_point, rb, ha)
    elseif itp_strategy == itpScatter
        W = Smoothed_kernel_function(Ktyp, reference_point, rb, input.hs[i])
    elseif itp_strategy == itpSymmetric
        W = T(0.5) * (Smoothed_kernel_function(Ktyp, reference_point, rb, ha) + Smoothed_kernel_function(Ktyp, reference_point, rb, input.hs[i]))
    end
    ∂xW :: T = zero(T)
    ∂yW :: T = zero(T)
    ∂zW :: T = zero(T)
    if itp_strategy == itpGather
        ∇W = Smoothed_gradient_kernel_function(Ktyp, reference_point, rb, ha)
        ∂xW = ∇W[1]
        ∂yW = ∇W[2]
        ∂zW = ∇W[3]
    elseif itp_strategy == itpScatter
        ∇W = Smoothed_gradient_kernel_function(Ktyp, reference_point, rb, input.hs[i])
        ∂xW = ∇W[1]
        ∂yW = ∇W[2]
        ∂zW = ∇W[3]
    elseif itp_strategy == itpSymmetric
        ∇Wa = Smoothed_gradient_kernel_function(Ktyp, reference_point, rb, ha)
        ∇Wb = Smoothed_gradient_kernel_function(Ktyp, reference_point, rb, input.hs[i])
        ∂xW = T(0.5) * (∇Wa[1] + ∇Wb[1])
        ∂yW = T(0.5) * (∇Wa[2] + ∇Wb[2])
        ∂zW = T(0.5) * (∇Wa[3] + ∇Wb[3])
    end

    # Gradient
    mb∂xW = mb * ∂xW
    mb∂yW = mb * ∂yW
    mb∂zW = mb * ∂zW

    ∇Af = mb∂xW * Axb + mb∂yW * Ayb + mb∂zW * Azb
    ∇Axb = mb∂xW
    ∇Ayb = mb∂yW
    ∇Azb = mb∂zW
    return ∇Af, ∇Axb, ∇Ayb, ∇Azb
end
@inline function _divergence_quantity_interpolate_kernel(input::ITPINPUT, reference_point::NTuple{3, T}, ha :: T, LBVH :: LinearBVH, Ax_column_idx :: Int64, Ay_column_idx :: Int64, Az_column_idx :: Int64, itp_strategy :: InterpolationStrategy = itpSymmetric) where {ITPINPUT <: AbstractInterpolationInput, T <: AbstractFloat}
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
    if root == 0
        nleaf = length(leaf_min[1])
        @inbounds for leaf_idx in 1:nleaf
            d2 = _dist2_to_leaf_aabb(leaf_min, leaf_max, reference_point, leaf_idx)
            if d2 <= radius2
                ########### Found a neighbor, do accumulation ###########
                ∇AfW, ∇AxbW, ∇AybW, ∇AzbW = _divergence_quantity_accumulation(input, reference_point, ha, Ax_column_idx, Ay_column_idx, Az_column_idx, itp_strategy, leaf_idx)
                ρ += _density_accumulation(input, reference_point, ha, itp_strategy, leaf_idx)
                Ax += _quantity_accumulation(input, reference_point, ha, Ax_column_idx, itp_strategy, leaf_idx)
                Ay += _quantity_accumulation(input, reference_point, ha, Ay_column_idx, itp_strategy, leaf_idx)
                Az += _quantity_accumulation(input, reference_point, ha, Az_column_idx, itp_strategy, leaf_idx)

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
        dist2_node = _dist2_to_node_aabb(node_min, node_max, reference_point, node)
        if dist2_node <= radius2
            if LL[node]
                @inbounds leaf_idx = L[node]
                d2 = _dist2_to_leaf_aabb(leaf_min, leaf_max, reference_point, leaf_idx)
                if d2 <= radius2
                    ########### Found a neighbor, do accumulation ###########
                    ∇AfW, ∇AxbW, ∇AybW, ∇AzbW = _divergence_quantity_accumulation(input, reference_point, ha, Ax_column_idx, Ay_column_idx, Az_column_idx, itp_strategy, leaf_idx)
                    ρ += _density_accumulation(input, reference_point, ha, itp_strategy, leaf_idx)
                    Ax += _quantity_accumulation(input, reference_point, ha, Ax_column_idx, itp_strategy, leaf_idx)
                    Ay += _quantity_accumulation(input, reference_point, ha, Ay_column_idx, itp_strategy, leaf_idx)
                    Az += _quantity_accumulation(input, reference_point, ha, Az_column_idx, itp_strategy, leaf_idx)

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
                d2 = _dist2_to_leaf_aabb(leaf_min, leaf_max, reference_point, leaf_idx)
                if d2 <= radius2
                    ########### Found a neighbor, do accumulation ###########
                    ∇AfW, ∇AxbW, ∇AybW, ∇AzbW = _divergence_quantity_accumulation(input, reference_point, ha, Ax_column_idx, Ay_column_idx, Az_column_idx, itp_strategy, leaf_idx)
                    ρ += _density_accumulation(input, reference_point, ha, itp_strategy, leaf_idx)
                    Ax += _quantity_accumulation(input, reference_point, ha, Ax_column_idx, itp_strategy, leaf_idx)
                    Ay += _quantity_accumulation(input, reference_point, ha, Ay_column_idx, itp_strategy, leaf_idx)
                    Az += _quantity_accumulation(input, reference_point, ha, Az_column_idx, itp_strategy, leaf_idx)

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

            node = _next_internal_node(node, L, R, LL, RR, node_parent)
        else
            node = _next_internal_node(node, L, R, LL, RR, node_parent)
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

"""
∇×A(r) = -(1/ρ(r))∑_b m_b*(A_b-A(r))×∇W(r-r_b)
       = -(1/ρ(r)) * ((∑_b m_b*A_b×∇W(r-r_b)) - A(r)×(∑_b m_b*∇W(r-r_b)))
       = -(1/ρ(r))*(∇×Af - ∇×Ab)
"""
# Single column curl value intepolation
@inline function _curl_quantity_accumulation(input :: ITPINPUT, reference_point::NTuple{3, T}, ha :: T, Ax_column_idx :: Int, Ay_column_idx :: Int, Az_column_idx :: Int, itp_strategy :: InterpolationStrategy, i :: Int) where {ITPINPUT <: AbstractInterpolationInput, T <: AbstractFloat}
    Ktyp = typeof(input.smoothed_kernel)
    mb = input.m[i]
    Axb = input.quant[Ax_column_idx][i]
    Ayb = input.quant[Ay_column_idx][i]
    Azb = input.quant[Az_column_idx][i]
    rb :: NTuple{3, T} = (input.x[i], input.y[i], input.z[i])

    W :: T = zero(T)
    if itp_strategy == itpGather
        W = Smoothed_kernel_function(Ktyp, reference_point, rb, ha)
    elseif itp_strategy == itpScatter
        W = Smoothed_kernel_function(Ktyp, reference_point, rb, input.hs[i])
    elseif itp_strategy == itpSymmetric
        W = T(0.5) * (Smoothed_kernel_function(Ktyp, reference_point, rb, ha) + Smoothed_kernel_function(Ktyp, reference_point, rb, input.hs[i]))
    end
    
    ∂xW :: T = zero(T)
    ∂yW :: T = zero(T)
    ∂zW :: T = zero(T)
    if itp_strategy == itpGather
        ∇W = Smoothed_gradient_kernel_function(Ktyp, reference_point, rb, ha)
        ∂xW = ∇W[1]
        ∂yW = ∇W[2]
        ∂zW = ∇W[3]
    elseif itp_strategy == itpScatter
        ∇W = Smoothed_gradient_kernel_function(Ktyp, reference_point, rb, input.hs[i])
        ∂xW = ∇W[1]
        ∂yW = ∇W[2]
        ∂zW = ∇W[3]
    elseif itp_strategy == itpSymmetric
        ∇Wa = Smoothed_gradient_kernel_function(Ktyp, reference_point, rb, ha)
        ∇Wb = Smoothed_gradient_kernel_function(Ktyp, reference_point, rb, input.hs[i])
        ∂xW = T(0.5) * (∇Wa[1] + ∇Wb[1])
        ∂yW = T(0.5) * (∇Wa[2] + ∇Wb[2])
        ∂zW = T(0.5) * (∇Wa[3] + ∇Wb[3])
    end

    # Gradient
    mb∂xW = mb * ∂xW
    mb∂yW = mb * ∂yW
    mb∂zW = mb * ∂zW

    ∇Axf += Ayb * mb∂zW -  Azb * mb∂yW
    ∇Ayf += Azb * mb∂xW -  Axb * mb∂zW
    ∇Azf += Axb * mb∂yW -  Ayb * mb∂xW
    m∂xW += mb∂xW
    m∂yW += mb∂yW
    m∂zW += mb∂zW
    return ∇Axf, ∇Ayf, ∇Azf, m∂xW, m∂yW, m∂zW
end
@inline function _curl_quantity_interpolate_kernel(input::ITPINPUT, reference_point::NTuple{3, T}, ha :: T, LBVH :: LinearBVH, Ax_column_idx :: Int64, Ay_column_idx :: Int64, Az_column_idx :: Int64, itp_strategy :: InterpolationStrategy = itpSymmetric) where {ITPINPUT <: AbstractInterpolationInput, T <: AbstractFloat}
    # Prepare for interpolation
    Ktyp = typeof(input.smoothed_kernel)
    Kvalid = KernelFunctionValid(Ktyp, T)

    # Initialize counter
    ∇Axf :: T = zero(T)
    ∇Ayf :: T = zero(T)
    ∇Azf :: T = zero(T)

    m∂xW :: T = zero(T)
    m∂yW :: T = zero(T)
    m∂zW :: T = zero(T)

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
    if root == 0
        nleaf = length(leaf_min[1])
        @inbounds for leaf_idx in 1:nleaf
            d2 = _dist2_to_leaf_aabb(leaf_min, leaf_max, reference_point, leaf_idx)
            if d2 <= radius2
                ########### Found a neighbor, do accumulation ###########
                ∇AxfW, ∇AyfW, ∇AzfW, m∂xWW, m∂yWW, m∂zWW = _curl_quantity_accumulation(input, reference_point, ha, Ax_column_idx, Ay_column_idx, Az_column_idx, itp_strategy, leaf_idx)
                ρ += _density_accumulation(input, reference_point, ha, itp_strategy, leaf_idx)
                Ax += _quantity_accumulation(input, reference_point, ha, Ax_column_idx, itp_strategy, leaf_idx)
                Ay += _quantity_accumulation(input, reference_point, ha, Ay_column_idx, itp_strategy, leaf_idx)
                Az += _quantity_accumulation(input, reference_point, ha, Az_column_idx, itp_strategy, leaf_idx)

                ∇Axf += ∇AxfW
                ∇Ayf += ∇AyfW
                ∇Azf += ∇AzfW
                m∂xW += m∂xWW
                m∂yW += m∂yWW
                m∂zW += m∂zWW
                mWlρ += _ShepardNormalization_accumulation(input, reference_point, ha, itp_strategy, leaf_idx)
                #########################################################
            end
        end
        if iszero(ρ)
            return (T(NaN), T(NaN), T(NaN))
        end

        # Shepard normalization
        Ax /= mWlρ
        Ay /= mWlρ
        Az /= mWlρ

        # Construct gradient
        ∇Axb = Ay * m∂zW - Az * m∂yW
        ∇Ayb = Az * m∂xW - Ax * m∂zW
        ∇Azb = Ax * m∂yW - Ay * m∂xW

        # Final result
        ∇Ax = -(∇Axf - ∇Axb)/ρ
        ∇Ay = -(∇Ayf - ∇Ayb)/ρ
        ∇Az = -(∇Azf - ∇Azb)/ρ

        return (∇Ax, ∇Ay, ∇Az)
    end

    # Start traversal
    node = root
    while node != 0
        dist2_node = _dist2_to_node_aabb(node_min, node_max, reference_point, node)
        if dist2_node <= radius2
            if LL[node]
                @inbounds leaf_idx = L[node]
                d2 = _dist2_to_leaf_aabb(leaf_min, leaf_max, reference_point, leaf_idx)
                if d2 <= radius2
                    ########### Found a neighbor, do accumulation ###########
                    ∇AxfW, ∇AyfW, ∇AzfW, m∂xWW, m∂yWW, m∂zWW = _curl_quantity_accumulation(input, reference_point, ha, Ax_column_idx, Ay_column_idx, Az_column_idx, itp_strategy, leaf_idx)
                    ρ += _density_accumulation(input, reference_point, ha, itp_strategy, leaf_idx)
                    Ax += _quantity_accumulation(input, reference_point, ha, Ax_column_idx, itp_strategy, leaf_idx)
                    Ay += _quantity_accumulation(input, reference_point, ha, Ay_column_idx, itp_strategy, leaf_idx)
                    Az += _quantity_accumulation(input, reference_point, ha, Az_column_idx, itp_strategy, leaf_idx)

                    ∇Axf += ∇AxfW
                    ∇Ayf += ∇AyfW
                    ∇Azf += ∇AzfW
                    m∂xW += m∂xWW
                    m∂yW += m∂yWW
                    m∂zW += m∂zWW
                    mWlρ += _ShepardNormalization_accumulation(input, reference_point, ha, itp_strategy, leaf_idx)
                    #########################################################
                end
            end
            if RR[node]
                @inbounds leaf_idx = R[node]
                d2 = _dist2_to_leaf_aabb(leaf_min, leaf_max, reference_point, leaf_idx)
                if d2 <= radius2
                    ########### Found a neighbor, do accumulation ###########
                    ∇AxfW, ∇AyfW, ∇AzfW, m∂xWW, m∂yWW, m∂zWW = _curl_quantity_accumulation(input, reference_point, ha, Ax_column_idx, Ay_column_idx, Az_column_idx, itp_strategy, leaf_idx)
                    ρ += _density_accumulation(input, reference_point, ha, itp_strategy, leaf_idx)
                    Ax += _quantity_accumulation(input, reference_point, ha, Ax_column_idx, itp_strategy, leaf_idx)
                    Ay += _quantity_accumulation(input, reference_point, ha, Ay_column_idx, itp_strategy, leaf_idx)
                    Az += _quantity_accumulation(input, reference_point, ha, Az_column_idx, itp_strategy, leaf_idx)

                    ∇Axf += ∇AxfW
                    ∇Ayf += ∇AyfW
                    ∇Azf += ∇AzfW
                    m∂xW += m∂xWW
                    m∂yW += m∂yWW
                    m∂zW += m∂zWW
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

            node = _next_internal_node(node, L, R, LL, RR, node_parent)
        else
            node = _next_internal_node(node, L, R, LL, RR, node_parent)
        end
    end
    if iszero(ρ)
        return (T(NaN), T(NaN), T(NaN))
    end


    # Shepard normalization
    Ax /= mWlρ
    Ay /= mWlρ
    Az /= mWlρ

    # Construct gradient
    ∇Axb = Ay * m∂zW - Az * m∂yW
    ∇Ayb = Az * m∂xW - Ax * m∂zW
    ∇Azb = Ax * m∂yW - Ay * m∂xW

    # Final result
    ∇Ax = -(∇Axf - ∇Axb)/ρ
    ∇Ay = -(∇Ayf - ∇Ayb)/ρ
    ∇Az = -(∇Azf - ∇Azb)/ρ

    return (∇Ax, ∇Ay, ∇Az)
end