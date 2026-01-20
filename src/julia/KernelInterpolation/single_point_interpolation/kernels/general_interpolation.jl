@inline function _general_quantity_interpolate_kernel(
                        input::InterpolationInput{T, V, Ktyp},
                        reference_point::NTuple{3,T},
                        ha::T,
                        LBVH::LinearBVH,
                        catalog::InterpolationCatalogConcise{N,G,D,C},
                        ::Type{itpGather}) :: Tuple{NTuple{N,T}, NTuple{G,NTuple{3,T}}, NTuple{D,T}, NTuple{C,NTuple{3,T}}} where {N, G, D, C, T<:AbstractFloat, V<:AbstractVector{T}, Ktyp<:AbstractSPHKernel}
    # Prepare for interpolation
    K = input.smoothed_kernel
    Kvalid = KernelFunctionValid(Ktyp, T)
    ShepardNormalization = catalog.scalar_snormalization
           
    # For aabb test
    radius = Kvalid * ha
    radius2 = radius * radius

    # Initialize counter
    ## Shepard Normalization
    S1 :: T = zero(T)

    ## Scalars
    scalars :: MVector{N, T} = zero(MVector{N, T})

    ## Gradients
    gradients_f :: MVector{G, SVector{3,T}} = MVector{G, SVector{3,T}}(ntuple(_ -> zero(SVector{3,T}), Val(G)))
    gradients_b :: MVector{G, SVector{3,T}} = MVector{G, SVector{3,T}}(ntuple(_ -> zero(SVector{3,T}), Val(G)))
    gradients_scalars :: MVector{G, T} = zero(MVector{G, T})                                                                  # Scalar that is used for estimating gradients

    ## Divergences
    divergences_f :: MVector{D, T} = zero(MVector{D, T})     
    divergences_b :: MVector{D, SVector{3,T}} = MVector{D, SVector{3,T}}(ntuple(_ -> zero(SVector{3,T}), Val(D)))
    divergences_scalars :: MVector{D, SVector{3,T}} = MVector{D, SVector{3,T}}(ntuple(_ -> zero(SVector{3,T}), Val(D)))       # Scalars that is used for estimating divergnece

    ## Curls
    curls_f :: MVector{C, SVector{3,T}} = MVector{C, SVector{3,T}}(ntuple(_ -> zero(SVector{3,T}), Val(C)))
    curls_b :: MVector{C, SVector{3,T}} = MVector{C, SVector{3,T}}(ntuple(_ -> zero(SVector{3,T}), Val(C)))
    curls_scalars :: MVector{C, SVector{3,T}} = MVector{C, SVector{3,T}}(ntuple(_ -> zero(SVector{3,T}), Val(C)))             # Scalars that is used for estimating curls

    # Traversal
    leaf_idx    :: Int = zero(Int)
    p2leaf_d2   :: T   = zero(T)

    NeighborSearch.@_lbvh_gather_traversal LBVH reference_point radius2 leaf_idx p2leaf_d2 begin
        ########### Found a neighbor, do accumulation ###########
        @inbounds begin
            rb = (input.x[leaf_idx], input.y[leaf_idx], input.z[leaf_idx])
            mb = input.m[leaf_idx]
            ρb = input.ρ[leaf_idx]

            # Shepard Normalization
            S1b = _ShepardNormalization_accumulation(reference_point, rb, mb, ρb, ha, K)
            S1 += S1b

            # Scalar interpolations
            @inbounds for j in 1:N
                slot = catalog.scalar_slots[j]
                Ab = input.quant[slot][leaf_idx]
                scalars[j] += _quantity_interpolate_accumulation(reference_point, rb, mb, ρb, Ab, ha, K)
            end

            # Gradient interpolations
            @inbounds for j in 1:G
                slot = catalog.grad_slots[j]
                if slot == 0
                    # Gradient of density
                    ∇ρxfW, ∇ρyfW, ∇ρzfW, ∇ρxbW, ∇ρybW, ∇ρzbW = _gradient_density_accumulation(reference_point, rb, mb, ρb, ha, K)
                    gradients_f[j] += SVector{3,T}(∇ρxfW, ∇ρyfW, ∇ρzfW)
                    gradients_b[j] += SVector{3,T}(∇ρxbW, ∇ρybW, ∇ρzbW)
                    gradients_scalars[j] += _density_accumulation(reference_point, rb, mb, ha, K)

                else
                    Ab = input.quant[slot][leaf_idx]
                    ∇AxfW, ∇AyfW, ∇AzfW, ∇AxbW, ∇AybW, ∇AzbW = _gradient_quantity_accumulation(reference_point, rb, mb, ρb, Ab, ha, K)
                    gradients_f[j] += SVector{3,T}(∇AxfW, ∇AyfW, ∇AzfW)
                    gradients_b[j] += SVector{3,T}(∇AxbW, ∇AybW, ∇AzbW)
                    gradients_scalars[j] += _quantity_interpolate_accumulation(reference_point, rb, mb, ρb, Ab, ha, K)
                end
            end

            # Divergence interpolations
            @inbounds for j in 1:D
                slot = catalog.div_slots[j]
                Ax_column_idx, Ay_column_idx, Az_column_idx = slot
                Axb = input.quant[Ax_column_idx][leaf_idx]
                Ayb = input.quant[Ay_column_idx][leaf_idx]
                Azb = input.quant[Az_column_idx][leaf_idx]
                ∇AfW, ∇AxbW, ∇AybW, ∇AzbW = _divergence_quantity_accumulation(reference_point, rb, mb, ρb, Axb, Ayb, Azb, ha, K)
                divergences_f[j] += ∇AfW
                divergences_b[j] += SVector{3,T}(∇AxbW, ∇AybW, ∇AzbW)
                Axa = _quantity_interpolate_accumulation(reference_point, rb, mb, ρb, Axb, ha, K)
                Aya = _quantity_interpolate_accumulation(reference_point, rb, mb, ρb, Ayb, ha, K)
                Aza = _quantity_interpolate_accumulation(reference_point, rb, mb, ρb, Azb, ha, K)
                divergences_scalars[j] += SVector{3,T}(Axa, Aya, Aza)
            end

            # Curl interpolations
            @inbounds for j in 1:C
                slot = catalog.curl_slots[j]
                Ax_column_idx, Ay_column_idx, Az_column_idx = slot
                Axb = input.quant[Ax_column_idx][leaf_idx]
                Ayb = input.quant[Ay_column_idx][leaf_idx]
                Azb = input.quant[Az_column_idx][leaf_idx]
                ∇AxfW, ∇AyfW, ∇AzfW, ∇AxbW, ∇AybW, ∇AzbW = _curl_quantity_accumulation(reference_point, rb, mb, ρb, Axb, Ayb, Azb, ha, K)
                curls_f[j] += SVector{3,T}(∇AxfW, ∇AyfW, ∇AzfW)
                curls_b[j] += SVector{3,T}(∇AxbW, ∇AybW, ∇AzbW)
                Axa = _quantity_interpolate_accumulation(reference_point, rb, mb, ρb, Axb, ha, K)
                Aya = _quantity_interpolate_accumulation(reference_point, rb, mb, ρb, Ayb, ha, K)
                Aza = _quantity_interpolate_accumulation(reference_point, rb, mb, ρb, Azb, ha, K)
                curls_scalars[j] += SVector{3,T}(Axa, Aya, Aza)
            end
        end
        #########################################################
    end

    # Preparing output
    if iszero(S1)
        output = (ntuple(_ -> T(NaN), Val(N)), ntuple(_ -> (T(NaN), T(NaN), T(NaN)), Val(G)), ntuple(i -> T(NaN), Val(D)), ntuple(i -> (T(NaN), T(NaN), T(NaN)), Val(C)))
        return output
    end

    # Shepard normalization
    invS1 = inv(S1)

    @inbounds for j in 1:N
        if ShepardNormalization[j]
            scalars[j] *= invS1
        end
    end

    # Construct gradients
    gradients_out = ntuple(Val(G)) do j
        @inbounds begin
            A = gradients_scalars[j] * invS1

            ∇Axf = gradients_f[j][1]
            ∇Ayf = gradients_f[j][2]
            ∇Azf = gradients_f[j][3]

            ∇Axb = gradients_b[j][1]
            ∇Ayb = gradients_b[j][2]
            ∇Azb = gradients_b[j][3]

            # Final result
            ∇Axb *= A
            ∇Ayb *= A
            ∇Azb *= A

            ∇Ax = ∇Axf - ∇Axb
            ∇Ay = ∇Ayf - ∇Ayb
            ∇Az = ∇Azf - ∇Azb

            (∇Ax, ∇Ay, ∇Az)
        end
    end

    # Construct divergences
    divergences_out = ntuple(Val(D)) do j 
        @inbounds begin
            Ax   = divergences_scalars[j][1] * invS1
            Ay   = divergences_scalars[j][2] * invS1
            Az   = divergences_scalars[j][3] * invS1

            ∇Af  = divergences_f[j]

            ∇Axb = divergences_b[j][1]
            ∇Ayb = divergences_b[j][2]
            ∇Azb = divergences_b[j][3]

            # Final result
            ∇Ab = Ax * ∇Axb + Ay * ∇Ayb + Az * ∇Azb
            ∇A = (∇Af - ∇Ab)

            ∇A
        end
    end

    # Construct curls
    curls_out = ntuple(Val(C)) do j 
        @inbounds begin
            Ax   = curls_scalars[j][1] * invS1
            Ay   = curls_scalars[j][2] * invS1
            Az   = curls_scalars[j][3] * invS1

            ∇Axf = curls_f[j][1]
            ∇Ayf = curls_f[j][2]
            ∇Azf = curls_f[j][3]

            mlρ∂xW = curls_b[j][1]
            mlρ∂yW = curls_b[j][2]
            mlρ∂zW = curls_b[j][3]

            # Final result
            ∇Axb = Ay * mlρ∂zW - Az * mlρ∂yW
            ∇Ayb = Az * mlρ∂xW - Ax * mlρ∂zW
            ∇Azb = Ax * mlρ∂yW - Ay * mlρ∂xW

            ∇Ax = -(∇Axf - ∇Axb)
            ∇Ay = -(∇Ayf - ∇Ayb)
            ∇Az = -(∇Azf - ∇Azb)

            (∇Ax, ∇Ay, ∇Az)
        end
    end

    scalars_out = ntuple(i -> scalars[i], Val(N))

    output = (scalars_out, gradients_out, divergences_out, curls_out)
    return output
end

@inline function _general_quantity_interpolate_kernel(
                        input::InterpolationInput{T, V, Ktyp},
                        reference_point::NTuple{3,T},
                        LBVH::LinearBVH,
                        catalog::InterpolationCatalogConcise{N,G,D,C},
                        ::Type{itpScatter}) :: Tuple{NTuple{N,T}, NTuple{G,NTuple{3,T}}, NTuple{D,T}, NTuple{C,NTuple{3,T}}} where {N, G, D, C, T<:AbstractFloat, V<:AbstractVector{T}, Ktyp<:AbstractSPHKernel}
    # Prepare for interpolation
    K = input.smoothed_kernel
    Kvalid = KernelFunctionValid(Ktyp, T)
    ShepardNormalization = catalog.scalar_snormalization
           
    # Initialize counter
    ## Shepard Normalization
    S1 :: T = zero(T)

    ## Scalars
    scalars :: MVector{N, T} = zero(MVector{N, T})

    ## Gradients
    gradients_f :: MVector{G, SVector{3,T}} = MVector{G, SVector{3,T}}(ntuple(_ -> zero(SVector{3,T}), Val(G)))
    gradients_b :: MVector{G, SVector{3,T}} = MVector{G, SVector{3,T}}(ntuple(_ -> zero(SVector{3,T}), Val(G)))
    gradients_scalars :: MVector{G, T} = zero(MVector{G, T})                                                                  # Scalar that is used for estimating gradients

    ## Divergences
    divergences_f :: MVector{D, T} = zero(MVector{D, T})     
    divergences_b :: MVector{D, SVector{3,T}} = MVector{D, SVector{3,T}}(ntuple(_ -> zero(SVector{3,T}), Val(D)))
    divergences_scalars :: MVector{D, SVector{3,T}} = MVector{D, SVector{3,T}}(ntuple(_ -> zero(SVector{3,T}), Val(D)))       # Scalars that is used for estimating divergnece

    ## Curls
    curls_f :: MVector{C, SVector{3,T}} = MVector{C, SVector{3,T}}(ntuple(_ -> zero(SVector{3,T}), Val(C)))
    curls_b :: MVector{C, SVector{3,T}} = MVector{C, SVector{3,T}}(ntuple(_ -> zero(SVector{3,T}), Val(C)))
    curls_scalars :: MVector{C, SVector{3,T}} = MVector{C, SVector{3,T}}(ntuple(_ -> zero(SVector{3,T}), Val(C)))             # Scalars that is used for estimating curls

    # Traversal
    leaf_idx    :: Int = zero(Int)
    p2leaf_d2   :: T   = zero(T)
    hb          :: T   = zero(T)

    NeighborSearch.@_lbvh_scatter_traversal LBVH reference_point Kvalid leaf_idx p2leaf_d2 hb begin
        ########### Found a neighbor, do accumulation ###########
        @inbounds begin
            rb = (input.x[leaf_idx], input.y[leaf_idx], input.z[leaf_idx])
            mb = input.m[leaf_idx]
            ρb = input.ρ[leaf_idx]

            # Shepard Normalization
            S1b = _ShepardNormalization_accumulation(reference_point, rb, mb, ρb, hb, K)
            S1 += S1b

            # Scalar interpolations
            @inbounds for j in 1:N
                slot = catalog.scalar_slots[j]
                Ab = input.quant[slot][leaf_idx]
                scalars[j] += _quantity_interpolate_accumulation(reference_point, rb, mb, ρb, Ab, hb, K)
            end

            # Gradient interpolations
            @inbounds for j in 1:G
                slot = catalog.grad_slots[j]
                if slot == 0
                    # Gradient of density
                    ∇ρxfW, ∇ρyfW, ∇ρzfW, ∇ρxbW, ∇ρybW, ∇ρzbW = _gradient_density_accumulation(reference_point, rb, mb, ρb, hb, K)
                    gradients_f[j] += SVector{3,T}(∇ρxfW, ∇ρyfW, ∇ρzfW)
                    gradients_b[j] += SVector{3,T}(∇ρxbW, ∇ρybW, ∇ρzbW)
                    gradients_scalars[j] += _density_accumulation(reference_point, rb, mb, hb, K)

                else
                    Ab = input.quant[slot][leaf_idx]
                    ∇AxfW, ∇AyfW, ∇AzfW, ∇AxbW, ∇AybW, ∇AzbW = _gradient_quantity_accumulation(reference_point, rb, mb, ρb, Ab, hb, K)
                    gradients_f[j] += SVector{3,T}(∇AxfW, ∇AyfW, ∇AzfW)
                    gradients_b[j] += SVector{3,T}(∇AxbW, ∇AybW, ∇AzbW)
                    gradients_scalars[j] += _quantity_interpolate_accumulation(reference_point, rb, mb, ρb, Ab, hb, K)
                end
            end

            # Divergence interpolations
            @inbounds for j in 1:D
                slot = catalog.div_slots[j]
                Ax_column_idx, Ay_column_idx, Az_column_idx = slot
                Axb = input.quant[Ax_column_idx][leaf_idx]
                Ayb = input.quant[Ay_column_idx][leaf_idx]
                Azb = input.quant[Az_column_idx][leaf_idx]
                ∇AfW, ∇AxbW, ∇AybW, ∇AzbW = _divergence_quantity_accumulation(reference_point, rb, mb, ρb, Axb, Ayb, Azb, hb, K)
                divergences_f[j] += ∇AfW
                divergences_b[j] += SVector{3,T}(∇AxbW, ∇AybW, ∇AzbW)
                Axa = _quantity_interpolate_accumulation(reference_point, rb, mb, ρb, Axb, hb, K)
                Aya = _quantity_interpolate_accumulation(reference_point, rb, mb, ρb, Ayb, hb, K)
                Aza = _quantity_interpolate_accumulation(reference_point, rb, mb, ρb, Azb, hb, K)
                divergences_scalars[j] += SVector{3,T}(Axa, Aya, Aza)
            end

            # Curl interpolations
            @inbounds for j in 1:C
                slot = catalog.curl_slots[j]
                Ax_column_idx, Ay_column_idx, Az_column_idx = slot
                Axb = input.quant[Ax_column_idx][leaf_idx]
                Ayb = input.quant[Ay_column_idx][leaf_idx]
                Azb = input.quant[Az_column_idx][leaf_idx]
                ∇AxfW, ∇AyfW, ∇AzfW, ∇AxbW, ∇AybW, ∇AzbW = _curl_quantity_accumulation(reference_point, rb, mb, ρb, Axb, Ayb, Azb, hb, K)
                curls_f[j] += SVector{3,T}(∇AxfW, ∇AyfW, ∇AzfW)
                curls_b[j] += SVector{3,T}(∇AxbW, ∇AybW, ∇AzbW)
                Axa = _quantity_interpolate_accumulation(reference_point, rb, mb, ρb, Axb, hb, K)
                Aya = _quantity_interpolate_accumulation(reference_point, rb, mb, ρb, Ayb, hb, K)
                Aza = _quantity_interpolate_accumulation(reference_point, rb, mb, ρb, Azb, hb, K)
                curls_scalars[j] += SVector{3,T}(Axa, Aya, Aza)
            end
        end
        #########################################################
        
    end

    # Preparing output
    if iszero(S1)
        output = (ntuple(_ -> T(NaN), Val(N)), ntuple(_ -> (T(NaN), T(NaN), T(NaN)), Val(G)), ntuple(i -> T(NaN), Val(D)), ntuple(i -> (T(NaN), T(NaN), T(NaN)), Val(C)))
        return output
    end

    # Shepard normalization
    invS1 = inv(S1)

    @inbounds for j in 1:N
        if ShepardNormalization[j]
            scalars[j] *= invS1
        end
    end

    # Construct gradients
    gradients_out = ntuple(Val(G)) do j
        @inbounds begin
            A = gradients_scalars[j] * invS1

            ∇Axf = gradients_f[j][1]
            ∇Ayf = gradients_f[j][2]
            ∇Azf = gradients_f[j][3]

            ∇Axb = gradients_b[j][1]
            ∇Ayb = gradients_b[j][2]
            ∇Azb = gradients_b[j][3]

            # Final result
            ∇Axb *= A
            ∇Ayb *= A
            ∇Azb *= A

            ∇Ax = ∇Axf - ∇Axb
            ∇Ay = ∇Ayf - ∇Ayb
            ∇Az = ∇Azf - ∇Azb

            (∇Ax, ∇Ay, ∇Az)
        end
    end

    # Construct divergences
    divergences_out = ntuple(Val(D)) do j 
        @inbounds begin
            Ax   = divergences_scalars[j][1] * invS1
            Ay   = divergences_scalars[j][2] * invS1
            Az   = divergences_scalars[j][3] * invS1

            ∇Af  = divergences_f[j]

            ∇Axb = divergences_b[j][1]
            ∇Ayb = divergences_b[j][2]
            ∇Azb = divergences_b[j][3]

            # Final result
            ∇Ab = Ax * ∇Axb + Ay * ∇Ayb + Az * ∇Azb
            ∇A = (∇Af - ∇Ab)

            ∇A
        end
    end

    # Construct curls
    curls_out = ntuple(Val(C)) do j 
        @inbounds begin
            Ax   = curls_scalars[j][1] * invS1
            Ay   = curls_scalars[j][2] * invS1
            Az   = curls_scalars[j][3] * invS1

            ∇Axf = curls_f[j][1]
            ∇Ayf = curls_f[j][2]
            ∇Azf = curls_f[j][3]

            mlρ∂xW = curls_b[j][1]
            mlρ∂yW = curls_b[j][2]
            mlρ∂zW = curls_b[j][3]

            # Final result
            ∇Axb = Ay * mlρ∂zW - Az * mlρ∂yW
            ∇Ayb = Az * mlρ∂xW - Ax * mlρ∂zW
            ∇Azb = Ax * mlρ∂yW - Ay * mlρ∂xW

            ∇Ax = -(∇Axf - ∇Axb)
            ∇Ay = -(∇Ayf - ∇Ayb)
            ∇Az = -(∇Azf - ∇Azb)

            (∇Ax, ∇Ay, ∇Az)
        end
    end

    scalars_out = ntuple(i -> scalars[i], Val(N))

    output = (scalars_out, gradients_out, divergences_out, curls_out)     
    return output 
end

@inline function _general_quantity_interpolate_kernel(
                        input::InterpolationInput{T, V, Ktyp},
                        reference_point::NTuple{3,T},
                        ha::T,
                        LBVH::LinearBVH,
                        catalog::InterpolationCatalogConcise{N,G,D,C},
                        ::Type{itpSymmetric}) :: Tuple{NTuple{N,T}, NTuple{G,NTuple{3,T}}, NTuple{D,T}, NTuple{C,NTuple{3,T}}} where {N, G, D, C, T<:AbstractFloat, V<:AbstractVector{T}, Ktyp<:AbstractSPHKernel}
    # Prepare for interpolation
    K = input.smoothed_kernel
    Kvalid = KernelFunctionValid(Ktyp, T)
    ShepardNormalization = catalog.scalar_snormalization
           
    # For aabb test
    radius = Kvalid * ha
    radius2 = radius * radius

    # Initialize counter
    ## Shepard Normalization
    S1 :: T = zero(T)
     
    ## Scalars
    scalars :: MVector{N, T} = zero(MVector{N, T})

    ## Gradients
    gradients_f :: MVector{G, SVector{3,T}} = MVector{G, SVector{3,T}}(ntuple(_ -> zero(SVector{3,T}), Val(G)))
    gradients_b :: MVector{G, SVector{3,T}} = MVector{G, SVector{3,T}}(ntuple(_ -> zero(SVector{3,T}), Val(G)))
    gradients_scalars :: MVector{G, T} = zero(MVector{G, T})                                                                  # Scalar that is used for estimating gradients

    ## Divergences
    divergences_f :: MVector{D, T} = zero(MVector{D, T})     
    divergences_b :: MVector{D, SVector{3,T}} = MVector{D, SVector{3,T}}(ntuple(_ -> zero(SVector{3,T}), Val(D)))
    divergences_scalars :: MVector{D, SVector{3,T}} = MVector{D, SVector{3,T}}(ntuple(_ -> zero(SVector{3,T}), Val(D)))       # Scalars that is used for estimating divergnece

    ## Curls
    curls_f :: MVector{C, SVector{3,T}} = MVector{C, SVector{3,T}}(ntuple(_ -> zero(SVector{3,T}), Val(C)))
    curls_b :: MVector{C, SVector{3,T}} = MVector{C, SVector{3,T}}(ntuple(_ -> zero(SVector{3,T}), Val(C)))
    curls_scalars :: MVector{C, SVector{3,T}} = MVector{C, SVector{3,T}}(ntuple(_ -> zero(SVector{3,T}), Val(C)))             # Scalars that is used for estimating curls

    # Traversal
    leaf_idx    :: Int = zero(Int)
    p2leaf_d2   :: T   = zero(T)
    hb          :: T   = zero(T)

    NeighborSearch.@_lbvh_symmetric_traversal LBVH reference_point Kvalid radius2 leaf_idx p2leaf_d2 hb begin
        ########### Found a neighbor, do accumulation ###########
        @inbounds begin
            rb = (input.x[leaf_idx], input.y[leaf_idx], input.z[leaf_idx])
            mb = input.m[leaf_idx]
            ρb = input.ρ[leaf_idx]

            # Shepard Normalization
            S1b = _ShepardNormalization_accumulation(reference_point, rb, mb, ρb, ha, hb, K)
            S1 += S1b
                

            # Scalar interpolations
            @inbounds for j in 1:N
                slot = catalog.scalar_slots[j]
                Ab = input.quant[slot][leaf_idx]
                scalars[j] += _quantity_interpolate_accumulation(reference_point, rb, mb, ρb, Ab, ha, hb, K)
            end

            # Gradient interpolations
            @inbounds for j in 1:G
                slot = catalog.grad_slots[j]
                if slot == 0
                    # Gradient of density
                    ∇ρxfW, ∇ρyfW, ∇ρzfW, ∇ρxbW, ∇ρybW, ∇ρzbW = _gradient_density_accumulation(reference_point, rb, mb, ρb, ha, hb, K)
                    gradients_f[j] += SVector{3,T}(∇ρxfW, ∇ρyfW, ∇ρzfW)
                    gradients_b[j] += SVector{3,T}(∇ρxbW, ∇ρybW, ∇ρzbW)
                    gradients_scalars[j] += _density_accumulation(reference_point, rb, mb, ha, hb, K)

                else
                    Ab = input.quant[slot][leaf_idx]
                    ∇AxfW, ∇AyfW, ∇AzfW, ∇AxbW, ∇AybW, ∇AzbW = _gradient_quantity_accumulation(reference_point, rb, mb, ρb, Ab, ha, hb, K)
                    gradients_f[j] += SVector{3,T}(∇AxfW, ∇AyfW, ∇AzfW)
                    gradients_b[j] += SVector{3,T}(∇AxbW, ∇AybW, ∇AzbW)
                    gradients_scalars[j] += _quantity_interpolate_accumulation(reference_point, rb, mb, ρb, Ab, ha, hb, K)
                end
            end

            # Divergence interpolations
            @inbounds for j in 1:D
                slot = catalog.div_slots[j]
                Ax_column_idx, Ay_column_idx, Az_column_idx = slot
                Axb = input.quant[Ax_column_idx][leaf_idx]
                Ayb = input.quant[Ay_column_idx][leaf_idx]
                Azb = input.quant[Az_column_idx][leaf_idx]
                ∇AfW, ∇AxbW, ∇AybW, ∇AzbW = _divergence_quantity_accumulation(reference_point, rb, mb, ρb, Axb, Ayb, Azb, ha, hb, K)
                divergences_f[j] += ∇AfW
                divergences_b[j] += SVector{3,T}(∇AxbW, ∇AybW, ∇AzbW)
                Axa = _quantity_interpolate_accumulation(reference_point, rb, mb, ρb, Axb, ha, hb, K)
                Aya = _quantity_interpolate_accumulation(reference_point, rb, mb, ρb, Ayb, ha, hb, K)
                Aza = _quantity_interpolate_accumulation(reference_point, rb, mb, ρb, Azb, ha, hb, K)
                divergences_scalars[j] += SVector{3,T}(Axa, Aya, Aza)
            end

            # Curl interpolations
            @inbounds for j in 1:C
                slot = catalog.curl_slots[j]
                Ax_column_idx, Ay_column_idx, Az_column_idx = slot
                Axb = input.quant[Ax_column_idx][leaf_idx]
                Ayb = input.quant[Ay_column_idx][leaf_idx]
                Azb = input.quant[Az_column_idx][leaf_idx]
                ∇AxfW, ∇AyfW, ∇AzfW, ∇AxbW, ∇AybW, ∇AzbW = _curl_quantity_accumulation(reference_point, rb, mb, ρb, Axb, Ayb, Azb, ha, hb, K)
                curls_f[j] += SVector{3,T}(∇AxfW, ∇AyfW, ∇AzfW)
                curls_b[j] += SVector{3,T}(∇AxbW, ∇AybW, ∇AzbW)
                Axa = _quantity_interpolate_accumulation(reference_point, rb, mb, ρb, Axb, ha, hb, K)
                Aya = _quantity_interpolate_accumulation(reference_point, rb, mb, ρb, Ayb, ha, hb, K)
                Aza = _quantity_interpolate_accumulation(reference_point, rb, mb, ρb, Azb, ha, hb, K)
                curls_scalars[j] += SVector{3,T}(Axa, Aya, Aza)
            end
        end
        #########################################################
    end

    # Preparing output
    if iszero(S1)
        output = (ntuple(_ -> T(NaN), Val(N)), ntuple(_ -> (T(NaN), T(NaN), T(NaN)), Val(G)), ntuple(i -> T(NaN), Val(D)), ntuple(i -> (T(NaN), T(NaN), T(NaN)), Val(C)))
        return output
    end

    # Shepard normalization
    invS1 = inv(S1)

    @inbounds for j in 1:N
        if ShepardNormalization[j]
            scalars[j] *= invS1
        end
    end

    # Construct gradients
    gradients_out = ntuple(Val(G)) do j
        @inbounds begin
            A = gradients_scalars[j] * invS1

            ∇Axf = gradients_f[j][1]
            ∇Ayf = gradients_f[j][2]
            ∇Azf = gradients_f[j][3]

            ∇Axb = gradients_b[j][1]
            ∇Ayb = gradients_b[j][2]
            ∇Azb = gradients_b[j][3]

            # Final result
            ∇Axb *= A
            ∇Ayb *= A
            ∇Azb *= A

            ∇Ax = ∇Axf - ∇Axb
            ∇Ay = ∇Ayf - ∇Ayb
            ∇Az = ∇Azf - ∇Azb

            (∇Ax, ∇Ay, ∇Az)
        end
    end

    # Construct divergences
    divergences_out = ntuple(Val(D)) do j 
        @inbounds begin
            Ax   = divergences_scalars[j][1] * invS1
            Ay   = divergences_scalars[j][2] * invS1
            Az   = divergences_scalars[j][3] * invS1

            ∇Af  = divergences_f[j]

            ∇Axb = divergences_b[j][1]
            ∇Ayb = divergences_b[j][2]
            ∇Azb = divergences_b[j][3]

            # Final result
            ∇Ab = Ax * ∇Axb + Ay * ∇Ayb + Az * ∇Azb
            ∇A = (∇Af - ∇Ab)

            ∇A
        end
    end

    # Construct curls
    curls_out = ntuple(Val(C)) do j 
        @inbounds begin
            Ax   = curls_scalars[j][1] * invS1
            Ay   = curls_scalars[j][2] * invS1
            Az   = curls_scalars[j][3] * invS1

            ∇Axf = curls_f[j][1]
            ∇Ayf = curls_f[j][2]
            ∇Azf = curls_f[j][3]

            mlρ∂xW = curls_b[j][1]
            mlρ∂yW = curls_b[j][2]
            mlρ∂zW = curls_b[j][3]

            # Final result
            ∇Axb = Ay * mlρ∂zW - Az * mlρ∂yW
            ∇Ayb = Az * mlρ∂xW - Ax * mlρ∂zW
            ∇Azb = Ax * mlρ∂yW - Ay * mlρ∂xW

            ∇Ax = -(∇Axf - ∇Axb)
            ∇Ay = -(∇Ayf - ∇Ayb)
            ∇Az = -(∇Azf - ∇Azb)

            (∇Ax, ∇Ay, ∇Az)
        end
    end

    scalars_out = ntuple(i -> scalars[i], Val(N))

    output = (scalars_out, gradients_out, divergences_out, curls_out)     
    return output 
end