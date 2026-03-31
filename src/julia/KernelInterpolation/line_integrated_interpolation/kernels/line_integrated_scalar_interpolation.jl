@inline function _line_integrated_density_kernel(input::ITPINPUT, origin::NTuple{3, T}, direction::NTuple{3, T}, ha :: T, LBVH :: LinearBVH, :: Type{itpGather} = itpGather) where {ITPINPUT <: AbstractInterpolationInput, T <: AbstractFloat}
    K = input.smoothed_kernel
    Ktyp = typeof(K)
    Kvalid = KernelFunctionValid(Ktyp, T)

    Sigma :: T = zero(T)

    radius = Kvalid * ha
    radius2 = radius * radius

    leaf_idx    :: Int = zero(Int)
    p2leaf_d2   :: T   = zero(T)

    NeighborSearch.@LBVH_gather_line_traversal LBVH origin direction radius2 leaf_idx p2leaf_d2 begin
        ########### Found a neighbor, do accumulation ###########
        @inbounds begin
            Δr = sqrt(p2leaf_d2)
            mb = input.m[leaf_idx]
            Sigma += _line_integrated_density_accumulation(Δr, mb, ha, K)
        end
        #########################################################
    end
    return Sigma
end

@inline function _line_integrated_density_kernel(input::ITPINPUT, origin::NTuple{3, T}, direction::NTuple{3, T}, LBVH :: LinearBVH, :: Type{itpScatter}) where {ITPINPUT <: AbstractInterpolationInput, T <: AbstractFloat}
    K = input.smoothed_kernel
    Ktyp = typeof(K)
    Kvalid = KernelFunctionValid(Ktyp, T)

    Sigma :: T = zero(T)

    leaf_idx    :: Int = zero(Int)
    p2leaf_d2   :: T   = zero(T)
    hb          :: T   = zero(T)

    NeighborSearch.@LBVH_scatter_line_traversal LBVH origin direction Kvalid leaf_idx p2leaf_d2 hb begin
        ########### Found a neighbor, do accumulation ###########
        @inbounds begin
            Δr = sqrt(p2leaf_d2)
            mb = input.m[leaf_idx]
            Sigma += _line_integrated_density_accumulation(Δr, mb, hb, K)
        end
        #########################################################
    end
    return Sigma
end

@inline function _line_integrated_density_kernel(input::ITPINPUT, origin::NTuple{3, T}, direction::NTuple{3, T}, ha :: T, LBVH :: LinearBVH, :: Type{itpSymmetric}) where {ITPINPUT <: AbstractInterpolationInput, T <: AbstractFloat}
    K = input.smoothed_kernel
    Ktyp = typeof(K)
    Kvalid = KernelFunctionValid(Ktyp, T)

    Sigma :: T = zero(T)

    radius = Kvalid * ha
    radius2 = radius * radius

    leaf_idx    :: Int = zero(Int)
    p2leaf_d2   :: T   = zero(T)
    hb          :: T   = zero(T)

    NeighborSearch.@LBVH_symmetric_line_traversal LBVH origin direction Kvalid radius2 leaf_idx p2leaf_d2 hb begin
        ########### Found a neighbor, do accumulation ###########
        @inbounds begin
            Δr = sqrt(p2leaf_d2)
            mb = input.m[leaf_idx]
            Sigma += _line_integrated_density_accumulation(Δr, mb, ha, hb, K)
        end
        #########################################################
    end
    return Sigma
end

## Multi-column line-integrated interpolation
@inline function _line_integrated_quantities_interpolate_kernel(input::ITPINPUT, origin::NTuple{3, T}, direction::NTuple{3, T}, ha :: T, LBVH :: LinearBVH, columns::NTuple{M,Int}, ShepardNormalization :: NTuple{M, Bool}, :: Type{itpGather} = itpGather) where {ITPINPUT <: AbstractInterpolationInput, T <: AbstractFloat, M}
    K = input.smoothed_kernel
    Ktyp = typeof(K)
    Kvalid = KernelFunctionValid(Ktyp, T)

    output :: MVector{M, T} = zero(MVector{M, T})
    S1 :: T = zero(T)

    radius = Kvalid * ha
    radius2 = radius * radius

    leaf_idx    :: Int = zero(Int)
    p2leaf_d2   :: T   = zero(T)

    NeighborSearch.@LBVH_gather_line_traversal LBVH origin direction radius2 leaf_idx p2leaf_d2 begin
        ########### Found a neighbor, do accumulation ###########
        @inbounds begin
            Δr = sqrt(p2leaf_d2)
            mb = input.m[leaf_idx]
            ρb = input.ρ[leaf_idx]

            S1b = _line_integrated_ShepardNormalization_accumulation(Δr, mb, ρb, ha, K)
            S1 += S1b

            @inbounds for j in 1:M
                column_idx = columns[j]
                Ab = input.quant[column_idx][leaf_idx]
                output[j] += _line_integrated_quantity_interpolate_accumulation(Δr, mb, ρb, Ab, ha, K)
            end
        end
        #########################################################
    end

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

@inline function _line_integrated_quantities_interpolate_kernel(input::ITPINPUT, origin::NTuple{3, T}, direction::NTuple{3, T}, LBVH :: LinearBVH, columns::NTuple{M,Int}, ShepardNormalization :: NTuple{M, Bool}, :: Type{itpScatter}) where {ITPINPUT <: AbstractInterpolationInput, T <: AbstractFloat, M}
    K = input.smoothed_kernel
    Ktyp = typeof(K)
    Kvalid = KernelFunctionValid(Ktyp, T)

    output :: MVector{M, T} = zero(MVector{M, T})
    S1 :: T = zero(T)

    leaf_idx    :: Int = zero(Int)
    p2leaf_d2   :: T   = zero(T)
    hb          :: T   = zero(T)

    NeighborSearch.@LBVH_scatter_line_traversal LBVH origin direction Kvalid leaf_idx p2leaf_d2 hb begin
        ########### Found a neighbor, do accumulation ###########
        @inbounds begin
            Δr = sqrt(p2leaf_d2)
            mb = input.m[leaf_idx]
            ρb = input.ρ[leaf_idx]

            S1b = _line_integrated_ShepardNormalization_accumulation(Δr, mb, ρb, hb, K)
            S1 += S1b

            @inbounds for j in 1:M
                column_idx = columns[j]
                Ab = input.quant[column_idx][leaf_idx]
                output[j] += _line_integrated_quantity_interpolate_accumulation(Δr, mb, ρb, Ab, hb, K)
            end
        end
        #########################################################
    end

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

@inline function _line_integrated_quantities_interpolate_kernel(input::ITPINPUT, origin::NTuple{3, T}, direction::NTuple{3, T}, ha :: T, LBVH :: LinearBVH, columns::NTuple{M,Int}, ShepardNormalization :: NTuple{M, Bool}, :: Type{itpSymmetric}) where {ITPINPUT <: AbstractInterpolationInput, T <: AbstractFloat, M}
    K = input.smoothed_kernel
    Ktyp = typeof(K)
    Kvalid = KernelFunctionValid(Ktyp, T)

    output :: MVector{M, T} = zero(MVector{M, T})
    S1 :: T = zero(T)

    radius = Kvalid * ha
    radius2 = radius * radius

    leaf_idx    :: Int = zero(Int)
    p2leaf_d2   :: T   = zero(T)
    hb          :: T   = zero(T)

    NeighborSearch.@LBVH_symmetric_line_traversal LBVH origin direction Kvalid radius2 leaf_idx p2leaf_d2 hb begin
        ########### Found a neighbor, do accumulation ###########
        @inbounds begin
            Δr = sqrt(p2leaf_d2)
            mb = input.m[leaf_idx]
            ρb = input.ρ[leaf_idx]

            S1b = _line_integrated_ShepardNormalization_accumulation(Δr, mb, ρb, ha, hb, K)
            S1 += S1b

            @inbounds for j in 1:M
                column_idx = columns[j]
                Ab = input.quant[column_idx][leaf_idx]
                output[j] += _line_integrated_quantity_interpolate_accumulation(Δr, mb, ρb, Ab, ha, hb, K)
            end
        end
        #########################################################
    end

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

@inline function _line_integrated_quantities_interpolate_kernel(input::ITPINPUT, origin::NTuple{3, T}, direction::NTuple{3, T}, ha :: T, LBVH :: LinearBVH, itp_strategy :: Type{ITPSTRATEGY} = itpSymmetric) where {ITPINPUT <: AbstractInterpolationInput, T <: AbstractFloat, ITPSTRATEGY <: AbstractInterpolationStrategy}
    val_len = Val(length(input.quant))
    columns = ntuple(identity, val_len)
    ShepardNormalization = ntuple(_ -> true, val_len)
    return _line_integrated_quantities_interpolate_kernel(input, origin, direction, ha, LBVH, columns, ShepardNormalization, itp_strategy)
end
