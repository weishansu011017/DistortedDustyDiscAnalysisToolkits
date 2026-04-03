@inline function _density_kernel(input::InterpolationInput{3, T}, reference_point::NTuple{3, T}, ha :: T, LBVH :: LinearBVH, :: Type{itpGather}) where {T <: AbstractFloat}
    # Prepare for interpolation
    K = input.smoothed_kernel
    Ktyp = typeof(K)
    Kvalid = KernelFunctionValid(Ktyp, T)

    # For aabb test
    radius = Kvalid * ha
    radius2 = radius * radius

    # Initialize counter
    rho :: T = zero(T)

    # Traversal
    leaf_idx    :: Int = zero(Int)
    p2leaf_d2   :: T   = zero(T)

    NeighborSearch.@LBVH_gather_point_traversal LBVH reference_point radius2 leaf_idx p2leaf_d2 begin
        ########### Found a neighbor, do accumulation ###########
        @inbounds begin
            Δr = sqrt(p2leaf_d2)
            mb = input.m[leaf_idx]
            rho += _density_accumulation(Δr, mb, ha, K)
        end
        #########################################################
    end
    return rho
end

@inline function _density_kernel(input::InterpolationInput{3, T}, reference_point::NTuple{3, T}, LBVH :: LinearBVH, :: Type{itpScatter}) where {T <: AbstractFloat}
    # Prepare for interpolation
    K = input.smoothed_kernel
    Ktyp = typeof(K)
    Kvalid = KernelFunctionValid(Ktyp, T)

    # Initialize counter
    rho :: T = zero(T)

    # Traversal
    leaf_idx    :: Int = zero(Int)
    p2leaf_d2   :: T   = zero(T)
    hb          :: T   = zero(T)

    NeighborSearch.@LBVH_scatter_point_traversal LBVH reference_point Kvalid leaf_idx p2leaf_d2 hb begin
        ########### Found a neighbor, do accumulation ###########
        @inbounds begin
            Δr = sqrt(p2leaf_d2)
            mb = input.m[leaf_idx]
            rho += _density_accumulation(Δr, mb, hb, K)
        end
        #########################################################
    end
    return rho
end

@inline function _density_kernel(input::InterpolationInput{3, T}, reference_point::NTuple{3, T}, ha :: T, LBVH :: LinearBVH, :: Type{itpSymmetric}) where {T <: AbstractFloat}
    # Prepare for interpolation
    K = input.smoothed_kernel
    Ktyp = typeof(K)
    Kvalid = KernelFunctionValid(Ktyp, T)

    # For aabb test
    radius = Kvalid * ha
    radius2 = radius * radius

    # Initialize counter
    rho :: T = zero(T)

    # Traversal
    leaf_idx    :: Int = zero(Int)
    p2leaf_d2   :: T   = zero(T)
    hb          :: T   = zero(T)

    NeighborSearch.@LBVH_symmetric_point_traversal LBVH reference_point Kvalid radius2 leaf_idx p2leaf_d2 hb begin
        ########### Found a neighbor, do accumulation ###########
        @inbounds begin
            Δr = sqrt(p2leaf_d2)
            mb = input.m[leaf_idx]
            rho += _density_accumulation(Δr, mb, ha, hb, K)
        end
        #########################################################
    end
    return rho
end

@inline function _number_density_kernel(input::InterpolationInput{3, T}, reference_point::NTuple{3, T}, ha :: T, LBVH :: LinearBVH, :: Type{itpGather}) where {T <: AbstractFloat}
    # Prepare for interpolation
    K = input.smoothed_kernel
    Ktyp = typeof(K)
    Kvalid = KernelFunctionValid(Ktyp, T)

    # For aabb test
    radius = Kvalid * ha
    radius2 = radius * radius

    # Initialize counter
    n :: T = zero(T)

    # Traversal
    leaf_idx    :: Int = zero(Int)
    p2leaf_d2   :: T   = zero(T)

    NeighborSearch.@LBVH_gather_point_traversal LBVH reference_point radius2 leaf_idx p2leaf_d2 begin
        ########### Found a neighbor, do accumulation ###########
        @inbounds begin
            Δr = sqrt(p2leaf_d2)
            n += _number_density_accumulation(Δr, ha, K)
        end
        #########################################################
    end
    return n
end

@inline function _number_density_kernel(input::InterpolationInput{3, T}, reference_point::NTuple{3, T}, LBVH :: LinearBVH, :: Type{itpScatter}) where {T <: AbstractFloat}
    # Prepare for interpolation
    K = input.smoothed_kernel
    Ktyp = typeof(K)
    Kvalid = KernelFunctionValid(Ktyp, T)

    # Initialize counter
    n :: T = zero(T)
    
    # Traversal
    leaf_idx    :: Int = zero(Int)
    p2leaf_d2   :: T   = zero(T)
    hb          :: T   = zero(T)

    NeighborSearch.@LBVH_scatter_point_traversal LBVH reference_point Kvalid leaf_idx p2leaf_d2 hb begin
        ########### Found a neighbor, do accumulation ###########
        @inbounds begin
            Δr = sqrt(p2leaf_d2)
            n += _number_density_accumulation(Δr, hb, K)
        end
        #########################################################
    end
    return n
end

@inline function _number_density_kernel(input::InterpolationInput{3, T}, reference_point::NTuple{3, T}, ha :: T, LBVH :: LinearBVH, :: Type{itpSymmetric}) where {T <: AbstractFloat}
    # Prepare for interpolation
    K = input.smoothed_kernel
    Ktyp = typeof(K)
    Kvalid = KernelFunctionValid(Ktyp, T)

    # For aabb test
    radius = Kvalid * ha
    radius2 = radius * radius

    # Initialize counter
    n :: T = zero(T)

    # Traversal
    leaf_idx    :: Int = zero(Int)
    p2leaf_d2   :: T   = zero(T)
    hb          :: T   = zero(T)

    NeighborSearch.@LBVH_symmetric_point_traversal LBVH reference_point Kvalid radius2 leaf_idx p2leaf_d2 hb begin
        ########### Found a neighbor, do accumulation ###########
        @inbounds begin
            Δr = sqrt(p2leaf_d2)
            n += _number_density_accumulation(Δr, ha, hb, K)
        end
        #########################################################
    end
    return n
end

@inline function _quantity_interpolate_kernel(input::InterpolationInput{3, T}, reference_point::NTuple{3, T}, ha :: T, LBVH :: LinearBVH, column_idx :: Int, ShepardNormalization :: Bool, :: Type{itpGather} = itpGather) where {T <: AbstractFloat}
    # Prepare for interpolation
    K = input.smoothed_kernel
    Ktyp = typeof(K)
    Kvalid = KernelFunctionValid(Ktyp, T)

    # Initialize counter
    A :: T = zero(T)
    S1 :: T = zero(T)
     

    radius = Kvalid * ha
    radius2 = radius * radius
    
    # Traversal
    leaf_idx    :: Int = zero(Int)
    p2leaf_d2   :: T   = zero(T)

    NeighborSearch.@LBVH_gather_point_traversal LBVH reference_point radius2 leaf_idx p2leaf_d2 begin
        ########### Found a neighbor, do accumulation ###########
        @inbounds begin
            Δr = sqrt(p2leaf_d2)
            mb = input.m[leaf_idx]
            ρb = input.ρ[leaf_idx]
            Ab = input.quant[column_idx][leaf_idx]
            A += _quantity_interpolate_accumulation(Δr, mb, ρb, Ab, ha, K)
            
            S1b = _ShepardNormalization_accumulation(Δr, mb, ρb, ha, K)
            S1 += S1b
             
        end
        #########################################################
    end
    # Shepard normalization
    if iszero(S1)
        return T(NaN)
    end
    if ShepardNormalization
        A /= S1
    end
     
    return A 
end

@inline function _quantity_interpolate_kernel(input::InterpolationInput{3, T}, reference_point::NTuple{3, T}, LBVH :: LinearBVH, column_idx :: Int, ShepardNormalization :: Bool, :: Type{itpScatter}) where {T <: AbstractFloat}
    K = input.smoothed_kernel
    Ktyp = typeof(K)
    Kvalid = KernelFunctionValid(Ktyp, T)

    A :: T = zero(T)
    S1 :: T = zero(T)

    # Traversal
    leaf_idx    :: Int = zero(Int)
    p2leaf_d2   :: T   = zero(T)
    hb          :: T   = zero(T)

    NeighborSearch.@LBVH_scatter_point_traversal LBVH reference_point Kvalid leaf_idx p2leaf_d2 hb begin
        ########### Found a neighbor, do accumulation ###########
        @inbounds begin
            Δr = sqrt(p2leaf_d2)
            mb = input.m[leaf_idx]
            ρb = input.ρ[leaf_idx]
            Ab = input.quant[column_idx][leaf_idx]
            A += _quantity_interpolate_accumulation(Δr, mb, ρb, Ab, hb, K)
            
            S1b = _ShepardNormalization_accumulation(Δr, mb, ρb, hb, K)
            S1 += S1b
             
        end
        #########################################################
    end
    # Shepard normalization
    if iszero(S1)
        return T(NaN)
    end
    if ShepardNormalization
        A /= S1
    end
     
    return A 
end

@inline function _quantity_interpolate_kernel(input::InterpolationInput{3, T}, reference_point::NTuple{3, T}, ha :: T, LBVH :: LinearBVH, column_idx :: Int, ShepardNormalization :: Bool, :: Type{itpSymmetric}) where {T <: AbstractFloat}
    K = input.smoothed_kernel
    Ktyp = typeof(K)
    Kvalid = KernelFunctionValid(Ktyp, T)

    A :: T = zero(T)
    S1 :: T = zero(T)
     

    radius = Kvalid * ha
    radius2 = radius * radius
    
    # Traversal
    leaf_idx    :: Int = zero(Int)
    p2leaf_d2   :: T   = zero(T)
    hb          :: T   = zero(T)

    NeighborSearch.@LBVH_symmetric_point_traversal LBVH reference_point Kvalid radius2 leaf_idx p2leaf_d2 hb begin
        ########### Found a neighbor, do accumulation ###########
        @inbounds begin
            Δr = sqrt(p2leaf_d2)
            mb = input.m[leaf_idx]
            ρb = input.ρ[leaf_idx]
            Ab = input.quant[column_idx][leaf_idx]
            A += _quantity_interpolate_accumulation(Δr, mb, ρb, Ab, ha, hb, K)
            
            S1b = _ShepardNormalization_accumulation(Δr, mb, ρb, ha, hb, K)
            S1 += S1b
             
        end
        #########################################################
    end
    # Shepard normalization
    if iszero(S1)
        return T(NaN)
    end
    if ShepardNormalization
        A /= S1
    end
     
    return A 
end

## Multi-column interpolation
@inline function _quantities_interpolate_kernel(input::InterpolationInput{3, T}, reference_point::NTuple{3, T}, ha :: T, LBVH :: LinearBVH, columns::NTuple{M,Int}, ShepardNormalization :: NTuple{M, Bool}, :: Type{itpGather} = itpGather) where {T <: AbstractFloat, M}
    # Prepare for interpolation
    K = input.smoothed_kernel
    Ktyp = typeof(K)
    Kvalid = KernelFunctionValid(Ktyp, T)

    # Initialize counter
    output :: MVector{M, T} = zero(MVector{M, T})
    S1 :: T = zero(T)
     
    
    radius = Kvalid * ha
    radius2 = radius * radius
    
    # Traversal
    leaf_idx    :: Int = zero(Int)
    p2leaf_d2   :: T   = zero(T)

    NeighborSearch.@LBVH_gather_point_traversal LBVH reference_point radius2 leaf_idx p2leaf_d2 begin
        ########### Found a neighbor, do accumulation ###########
        @inbounds begin
            Δr = sqrt(p2leaf_d2)
            mb = input.m[leaf_idx]
            ρb = input.ρ[leaf_idx]
            
            S1b = _ShepardNormalization_accumulation(Δr, mb, ρb, ha, K)
            S1 += S1b
             
            @inbounds for j in 1:M
                column_idx = columns[j]
                Ab = input.quant[column_idx][leaf_idx]
                output[j] += _quantity_interpolate_accumulation(Δr, mb, ρb, Ab, ha, K)
            end
        end
        #########################################################
    end
    # Shepard normalization
    if iszero(S1)
        return ntuple(_ -> T(NaN), Val(M))
    end

    invS1 = inv(S1)
    @inbounds for j in 1:M
        if ShepardNormalization[j]
            output[j] *= invS1
        end
    end
     
    return NTuple{M, T}(output) 
end

@inline function _quantities_interpolate_kernel(input::InterpolationInput{3, T}, reference_point::NTuple{3, T}, LBVH :: LinearBVH, columns::NTuple{M,Int}, ShepardNormalization :: NTuple{M, Bool}, :: Type{itpScatter}) where {T <: AbstractFloat, M}
    K = input.smoothed_kernel
    Ktyp = typeof(K)
    Kvalid = KernelFunctionValid(Ktyp, T)

    output :: MVector{M, T} = zero(MVector{M, T})
    S1 :: T = zero(T)

    
    # Traversal
    leaf_idx    :: Int = zero(Int)
    p2leaf_d2   :: T   = zero(T)
    hb          :: T   = zero(T)

    NeighborSearch.@LBVH_scatter_point_traversal LBVH reference_point Kvalid leaf_idx p2leaf_d2 hb begin
        ########### Found a neighbor, do accumulation ###########
        @inbounds begin
            Δr = sqrt(p2leaf_d2)
            mb = input.m[leaf_idx]
            ρb = input.ρ[leaf_idx]
            
            S1b = _ShepardNormalization_accumulation(Δr, mb, ρb, hb, K)
            S1 += S1b
             
            @inbounds for j in 1:M
                column_idx = columns[j]
                Ab = input.quant[column_idx][leaf_idx]
                output[j] += _quantity_interpolate_accumulation(Δr, mb, ρb, Ab, hb, K)
            end
        end
        #########################################################
    end
    # Shepard normalization
    if iszero(S1)
        return ntuple(_ -> T(NaN), Val(M))
    end

    invS1 = inv(S1)
    @inbounds for j in 1:M
        if ShepardNormalization[j]
            output[j] *= invS1
        end
    end
     
    return NTuple{M, T}(output) 
end

@inline function _quantities_interpolate_kernel(input::InterpolationInput{3, T}, reference_point::NTuple{3, T}, ha :: T, LBVH :: LinearBVH, columns::NTuple{M,Int}, ShepardNormalization :: NTuple{M, Bool}, :: Type{itpSymmetric}) where {T <: AbstractFloat, M}
    K = input.smoothed_kernel
    Ktyp = typeof(K)
    Kvalid = KernelFunctionValid(Ktyp, T)

    radius = Kvalid * ha
    radius2 = radius * radius

    output :: MVector{M, T} = zero(MVector{M, T})
    S1 :: T = zero(T)
    
    # Traversal
    leaf_idx    :: Int = zero(Int)
    p2leaf_d2   :: T   = zero(T)
    hb          :: T   = zero(T)

    NeighborSearch.@LBVH_symmetric_point_traversal LBVH reference_point Kvalid radius2 leaf_idx p2leaf_d2 hb begin
        ########### Found a neighbor, do accumulation ###########
        @inbounds begin
            Δr = sqrt(p2leaf_d2)
            mb = input.m[leaf_idx]
            ρb = input.ρ[leaf_idx]
            
            S1b = _ShepardNormalization_accumulation(Δr, mb, ρb, ha, hb, K)
            S1 += S1b
             
            @inbounds for j in 1:M
                column_idx = columns[j]
                Ab = input.quant[column_idx][leaf_idx]
                output[j] += _quantity_interpolate_accumulation(Δr, mb, ρb, Ab, ha, hb, K)
            end
        end
        #########################################################
    end
    # Shepard normalization
    if iszero(S1)
        return ntuple(_ -> T(NaN), Val(M))
    end

    invS1 = inv(S1)
    @inbounds for j in 1:M
        if ShepardNormalization[j]
            output[j] *= invS1
        end
    end
     
    return NTuple{M, T}(output) 
end

@inline function _quantities_interpolate_kernel(input::InterpolationInput{3, T}, reference_point::NTuple{3, T}, ha :: T, LBVH :: LinearBVH, itp_strategy :: Type{ITPSTRATEGY} = itpSymmetric) where {T <: AbstractFloat, ITPSTRATEGY <: AbstractInterpolationStrategy}
    len = length(input.quant)
    columns = ntuple(identity, len)
    ShepardNormalization = ntuple(_ -> true, len)
    return _quantities_interpolate_kernel(input, reference_point, ha, LBVH, columns, ShepardNormalization, itp_strategy)
end
