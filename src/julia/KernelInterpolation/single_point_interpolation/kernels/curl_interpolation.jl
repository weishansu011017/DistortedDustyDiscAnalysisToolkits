@inline function _curl_quantity_interpolate_kernel(input::ITPINPUT, reference_point::NTuple{3, T}, ha :: T, LBVH :: LinearBVH, Ax_column_idx :: Int, Ay_column_idx :: Int, Az_column_idx :: Int, :: Type{itpGather}) where {ITPINPUT <: AbstractInterpolationInput, T <: AbstractFloat}
    K = input.smoothed_kernel
    Ktyp = typeof(K)
    Kvalid = KernelFunctionValid(Ktyp, T)

    radius = Kvalid * ha
    radius2 = radius * radius

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
     

    
    # Traversal
    leaf_idx    :: Int = zero(Int)
    p2leaf_d2   :: T   = zero(T)

    NeighborSearch.@LBVH_gather_traversal LBVH reference_point radius2 leaf_idx p2leaf_d2 begin
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
             
        end
        #########################################################
    end

    if iszero(S1)
        return (T(NaN), T(NaN), T(NaN)), NaN32
    end

    Ax /= S1
    Ay /= S1
    Az /= S1

    ∇Axb = Ay * mlρ∂zW - Az * mlρ∂yW
    ∇Ayb = Az * mlρ∂xW - Ax * mlρ∂zW
    ∇Azb = Ax * mlρ∂yW - Ay * mlρ∂xW

    ∇Ax = -(∇Axf - ∇Axb)
    ∇Ay = -(∇Ayf - ∇Ayb)
    ∇Az = -(∇Azf - ∇Azb)

     
    return (∇Ax, ∇Ay, ∇Az) 
end

@inline function _curl_quantity_interpolate_kernel(input::ITPINPUT, reference_point::NTuple{3, T}, LBVH :: LinearBVH, Ax_column_idx :: Int, Ay_column_idx :: Int, Az_column_idx :: Int, :: Type{itpScatter}) where {ITPINPUT <: AbstractInterpolationInput, T <: AbstractFloat}
    K = input.smoothed_kernel
    Ktyp = typeof(K)
    Kvalid = KernelFunctionValid(Ktyp, T)

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
     

    
    # Traversal
    leaf_idx    :: Int = zero(Int)
    p2leaf_d2   :: T   = zero(T)
    hb          :: T   = zero(T)

    NeighborSearch.@LBVH_scatter_traversal LBVH reference_point Kvalid leaf_idx p2leaf_d2 hb begin
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
             
        end
        #########################################################
    end

    if iszero(S1)
        return (T(NaN), T(NaN), T(NaN)), NaN32
    end

    Ax /= S1
    Ay /= S1
    Az /= S1

    ∇Axb = Ay * mlρ∂zW - Az * mlρ∂yW
    ∇Ayb = Az * mlρ∂xW - Ax * mlρ∂zW
    ∇Azb = Ax * mlρ∂yW - Ay * mlρ∂xW

    ∇Ax = -(∇Axf - ∇Axb)
    ∇Ay = -(∇Ayf - ∇Ayb)
    ∇Az = -(∇Azf - ∇Azb)

     
    return (∇Ax, ∇Ay, ∇Az) 
end

@inline function _curl_quantity_interpolate_kernel(input::ITPINPUT, reference_point::NTuple{3, T}, ha :: T, LBVH :: LinearBVH, Ax_column_idx :: Int, Ay_column_idx :: Int, Az_column_idx :: Int, :: Type{itpSymmetric}) where {ITPINPUT <: AbstractInterpolationInput, T <: AbstractFloat}
    K = input.smoothed_kernel
    Ktyp = typeof(K)
    Kvalid = KernelFunctionValid(Ktyp, T)

    radius = Kvalid * ha
    radius2 = radius * radius

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
     

    
    # Traversal
    leaf_idx    :: Int = zero(Int)
    p2leaf_d2   :: T   = zero(T)
    hb          :: T   = zero(T)

    NeighborSearch.@LBVH_symmetric_traversal LBVH reference_point Kvalid radius2 leaf_idx p2leaf_d2 hb begin
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
             
        end
        #########################################################
    end

    if iszero(S1)
        return (T(NaN), T(NaN), T(NaN)), NaN32
    end

    Ax /= S1
    Ay /= S1
    Az /= S1

    ∇Axb = Ay * mlρ∂zW - Az * mlρ∂yW
    ∇Ayb = Az * mlρ∂xW - Ax * mlρ∂zW
    ∇Azb = Ax * mlρ∂yW - Ay * mlρ∂xW

    ∇Ax = -(∇Axf - ∇Axb)
    ∇Ay = -(∇Ayf - ∇Ayb)
    ∇Az = -(∇Azf - ∇Azb)

     
    return (∇Ax, ∇Ay, ∇Az) 
end