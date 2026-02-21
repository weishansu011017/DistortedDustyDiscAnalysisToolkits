@inline function _LOS_density_kernel(input::ITPINPUT, reference_point::NTuple{2, T}, ha :: T, LBVH :: LinearBVH, :: Type{itpGather} = itpGather) where {ITPINPUT <: AbstractInterpolationInput, T <: AbstractFloat}
    # Prepare for interpolation
    K = input.smoothed_kernel
    Ktyp = typeof(K)
    Kvalid = KernelFunctionValid(Ktyp, T)

    # Initialize counter
    Sigma :: T = zero(T)
    
    radius = Kvalid * ha
    radius2 = radius * radius
    
    # Traversal
    leaf_idx    :: Int = zero(Int)
    p2leaf_d2   :: T   = zero(T)

    NeighborSearch.@LBVH_gather_traversal LBVH reference_point radius2 leaf_idx p2leaf_d2 begin
        ########### Found a neighbor, do accumulation ###########
        @inbounds begin
            rb = (input.x[leaf_idx], input.y[leaf_idx])
            mb = input.m[leaf_idx]
            Sigma += _LOS_density_accumulation(reference_point, rb, mb, ha, K)
        end
        #########################################################
    end
    return Sigma
end

@inline function _LOS_density_kernel(input::ITPINPUT, reference_point::NTuple{2, T}, LBVH :: LinearBVH, :: Type{itpScatter}) where {ITPINPUT <: AbstractInterpolationInput, T <: AbstractFloat}
    K = input.smoothed_kernel
    Ktyp = typeof(K)
    Kvalid = KernelFunctionValid(Ktyp, T)

    Sigma = zero(T)

    
    # Traversal
    leaf_idx    :: Int = zero(Int)
    p2leaf_d2   :: T   = zero(T)
    hb          :: T   = zero(T)

    NeighborSearch.@LBVH_scatter_traversal LBVH reference_point Kvalid leaf_idx p2leaf_d2 hb begin
        ########### Found a neighbor, do accumulation ###########
        @inbounds begin
            rb = (input.x[leaf_idx], input.y[leaf_idx])
            mb = input.m[leaf_idx]
            Sigma += _LOS_density_accumulation(reference_point, rb, mb, hb, K)
        end
        #########################################################
    end
    return Sigma
end

@inline function _LOS_density_kernel(input::ITPINPUT, reference_point::NTuple{2, T}, ha :: T, LBVH :: LinearBVH, :: Type{itpSymmetric}) where {ITPINPUT <: AbstractInterpolationInput, T <: AbstractFloat}
    K = input.smoothed_kernel
    Ktyp = typeof(K)
    Kvalid = KernelFunctionValid(Ktyp, T)

    Sigma = zero(T)

    radius = Kvalid * ha
    radius2 = radius * radius
    
    # Traversal
    leaf_idx    :: Int = zero(Int)
    p2leaf_d2   :: T   = zero(T)
    hb          :: T   = zero(T)

    NeighborSearch.@LBVH_symmetric_traversal LBVH reference_point Kvalid radius2 leaf_idx p2leaf_d2 hb begin
        ########### Found a neighbor, do accumulation ###########
        @inbounds begin
            rb = (input.x[leaf_idx], input.y[leaf_idx])
            mb = input.m[leaf_idx]
            Sigma += _LOS_density_accumulation(reference_point, rb, mb, ha, hb, K)
        end
        #########################################################
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
    output :: MVector{M, T} = zero(MVector{M, T})
    S1 :: T = zero(T)
     
    radius = Kvalid * ha
    radius2 = radius * radius
    
    # Traversal
    leaf_idx    :: Int = zero(Int)
    p2leaf_d2   :: T   = zero(T)

    NeighborSearch.@LBVH_gather_traversal LBVH reference_point radius2 leaf_idx p2leaf_d2 begin
        ########### Found a neighbor, do accumulation ###########
        @inbounds begin
            rb = (input.x[leaf_idx], input.y[leaf_idx])
            mb = input.m[leaf_idx]
            ρb = input.ρ[leaf_idx]
            
            S1b = _LOS_ShepardNormalization_accumulation(reference_point, rb, mb, ρb, ha, K)
            S1 += S1b
             
            @inbounds for j in 1:M
                column_idx = columns[j]
                Ab = input.quant[column_idx][leaf_idx]
                output[j] += _LOS_quantity_interpolate_accumulation(reference_point, rb, mb, ρb, Ab, ha, K)
            end
        end
        #########################################################
    end
    if iszero(S1)
        return ntuple(_ -> T(NaN), Val(M)), NaN32
    end

    invS1 = inv(S1)
    @inbounds for j in 1:M
        if ShepardNormalization[j]
            output[j] *= invS1
        end
    end
     
    return NTuple{M, T}(output) 
end

@inline function _LOS_quantities_interpolate_kernel(input::ITPINPUT, reference_point::NTuple{2, T}, LBVH :: LinearBVH, columns::NTuple{M,Int}, ShepardNormalization :: NTuple{M, Bool}, :: Type{itpScatter}) where {ITPINPUT <: AbstractInterpolationInput, T <: AbstractFloat, M}
    K = input.smoothed_kernel
    Ktyp = typeof(K)
    Kvalid = KernelFunctionValid(Ktyp, T)

    output :: MVector{M, T} = zero(MVector{M, T})
    S1 :: T = zero(T)
     
    
    # Traversal
    leaf_idx    :: Int = zero(Int)
    p2leaf_d2   :: T   = zero(T)
    hb          :: T   = zero(T)

    NeighborSearch.@LBVH_scatter_traversal LBVH reference_point Kvalid leaf_idx p2leaf_d2 hb begin
        ########### Found a neighbor, do accumulation ###########
        @inbounds begin
            rb = (input.x[leaf_idx], input.y[leaf_idx])
            mb = input.m[leaf_idx]
            ρb = input.ρ[leaf_idx]
            
            S1b = _LOS_ShepardNormalization_accumulation(reference_point, rb, mb, ρb, hb, K)
            S1 += S1b
             
            @inbounds for j in 1:M
                column_idx = columns[j]
                Ab = input.quant[column_idx][leaf_idx]
                output[j] += _LOS_quantity_interpolate_accumulation(reference_point, rb, mb, ρb, Ab, hb, K)
            end
        end
        #########################################################
    end
    if iszero(S1)
        return ntuple(_ -> T(NaN), Val(M)), NaN32
    end

    invS1 = inv(S1)
    @inbounds for j in 1:M
        if ShepardNormalization[j]
            output[j] *= invS1
        end
    end
     
    return NTuple{M, T}(output) 
end

@inline function _LOS_quantities_interpolate_kernel(input::ITPINPUT, reference_point::NTuple{2, T}, ha :: T, LBVH :: LinearBVH, columns::NTuple{M,Int}, ShepardNormalization :: NTuple{M, Bool}, :: Type{itpSymmetric}) where {ITPINPUT <: AbstractInterpolationInput, T <: AbstractFloat, M}
    K = input.smoothed_kernel
    Ktyp = typeof(K)
    Kvalid = KernelFunctionValid(Ktyp, T)

    output :: MVector{M, T} = zero(MVector{M, T})
    S1 :: T = zero(T)
     
    radius = Kvalid * ha
    radius2 = radius * radius
    
    # Traversal
    leaf_idx    :: Int = zero(Int)
    p2leaf_d2   :: T   = zero(T)
    hb          :: T   = zero(T)

    NeighborSearch.@LBVH_symmetric_traversal LBVH reference_point Kvalid radius2 leaf_idx p2leaf_d2 hb begin
        ########### Found a neighbor, do accumulation ###########
        @inbounds begin
            rb = (input.x[leaf_idx], input.y[leaf_idx])
            mb = input.m[leaf_idx]
            ρb = input.ρ[leaf_idx]
            
            S1b = _LOS_ShepardNormalization_accumulation(reference_point, rb, mb, ρb, ha, hb, K)
            S1 += S1b
             
            @inbounds for j in 1:M
                column_idx = columns[j]
                Ab = input.quant[column_idx][leaf_idx]
                output[j] += _LOS_quantity_interpolate_accumulation(reference_point, rb, mb, ρb, Ab, ha, hb, K)
            end
        end
        #########################################################
    end
    if iszero(S1)
        return ntuple(_ -> T(NaN), Val(M)), NaN32
    end

    invS1 = inv(S1)
    @inbounds for j in 1:M
        if ShepardNormalization[j]
            output[j] *= invS1
        end
    end
     
    return NTuple{M, T}(output) 
end

@inline function _LOS_quantities_interpolate_kernel(input::ITPINPUT, reference_point::NTuple{2, T}, ha :: T, LBVH :: LinearBVH, itp_strategy :: Type{ITPSTRATEGY} = itpSymmetric) where {ITPINPUT <: AbstractInterpolationInput, T <: AbstractFloat, ITPSTRATEGY <: AbstractInterpolationStrategy}
    val_len = Val(length(input.quant))
    columns = ntuple(identity, val_len)
    ShepardNormalization = ntuple(_ -> true, val_len)
    return _LOS_quantities_interpolate_kernel(input, reference_point, ha, LBVH, columns, ShepardNormalization, itp_strategy)
end
