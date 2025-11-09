# Determine interpolation type
@enum InterpolationStrategy begin
    itpGather
    itpScatter
    itpSymmetric
end

## Density
@inline function _density_kernel(input::ITPINPUT, reference_point::NTuple{3, T}, ha :: T, neighbors :: NeighborSelection, itp_strategy :: InterpolationStrategy = itpSymmetric) where {ITPINPUT <: AbstractInterpolationInput, T <: AbstractFloat}
    # Return 0.0 if no particle in the data
    neighbor_indices = neighbors.pool
    Npart :: Int64 = neighbors.count
    if Npart == 0
        return zero(T)
    end

    # Prepare for interpolation
    xs = input.x
    ys = input.y
    zs = input.z
    ms = input.m
    hs = input.h
    Ktyp = input.smoothed_kernel

    # Initialize counter
    rho :: T = zero(T)

    @inbounds for k in 1:Npart
        i = neighbor_indices[k]
        rb :: NTuple{3, T} = (xs[i], ys[i], zs[i])
        W :: T = zero(T)
        if itp_strategy == itpGather
            W = Smoothed_kernel_function(Ktyp, reference_point, rb, ha)
        elseif itp_strategy == itpScatter
            W = Smoothed_kernel_function(Ktyp, reference_point, rb, hs[i])
        elseif itp_strategy == itpSymmetric
            W = T(0.5) * (Smoothed_kernel_function(Ktyp, reference_point, rb, ha) + Smoothed_kernel_function(Ktyp, reference_point, rb, hs[i]))
        end
        rho += ms[i] * W
    end
    return rho
end

## Number density
@inline function _number_density_kernel(input::ITPINPUT, reference_point::NTuple{3, T}, ha :: T, neighbors :: NeighborSelection, itp_strategy :: InterpolationStrategy = itpSymmetric) where {ITPINPUT <: AbstractInterpolationInput, T <: AbstractFloat}
    # Return 0.0 if no particle in the data
    neighbor_indices = neighbors.pool
    Npart :: Int64 = neighbors.count
    if Npart == 0
    return zero(T)
    end

    # Prepare for interpolation
    # Known quantities
    xs = input.x
    ys = input.y
    zs = input.z
    hs = input.h
    Ktyp = input.smoothed_kernel

    # Initialize counter
    n :: T = zero(T)

    @inbounds for k in 1:Npart
        i = neighbor_indices[k]
        rb :: NTuple{3, T} = (xs[i], ys[i], zs[i])
        W :: T = zero(T)
        if itp_strategy == itpGather
            W = Smoothed_kernel_function(Ktyp, reference_point, rb, ha)
        elseif itp_strategy == itpScatter
            W = Smoothed_kernel_function(Ktyp, reference_point, rb, hs[i])
        elseif itp_strategy == itpSymmetric
            W = T(0.5) * (Smoothed_kernel_function(Ktyp, reference_point, rb, ha) + Smoothed_kernel_function(Ktyp, reference_point, rb, hs[i]))
        end
        n += W
    end
    return n
end

## Single quantity intepolation
@inline function _quantity_interpolate_kernel(input::ITPINPUT, reference_point::NTuple{3, T}, ha :: T, neighbors :: NeighborSelection, column_idx :: Int64, itp_strategy :: InterpolationStrategy = itpSymmetric) where {ITPINPUT <: AbstractInterpolationInput, T <: AbstractFloat}
    # Return 0.0 if no particle in the data
    neighbor_indices = neighbors.pool
    Npart :: Int64 = neighbors.count
    if Npart == 0
        return zero(T)
    end

    if column_idx == 0
        return _density_kernel(input, reference_point, ha, neighbors, itp_strategy)
    end
    # Prepare for interpolation
    # Known quantities
    xs = input.x
    ys = input.y
    zs = input.z
    ms = input.m
    hs = input.h
    ρs = input.ρ
    As = input.quant[column_idx]
    Ktyp = input.smoothed_kernel

    # Initialize counter
    A :: T = zero(T)
    mWlρ :: T = zero(T)

    @inbounds for k in 1:Npart
        i = neighbor_indices[k]
        mb = ms[i]
        ρb = ρs[i]
        Ab = As[i]
        rb :: NTuple{3, T} = (xs[i], ys[i], zs[i])
        W :: T = zero(T)
        if itp_strategy == itpGather
            W = Smoothed_kernel_function(Ktyp, reference_point, rb, ha)
        elseif itp_strategy == itpScatter
            W = Smoothed_kernel_function(Ktyp, reference_point, rb, hs[i])
        elseif itp_strategy == itpSymmetric
            W = T(0.5) * (Smoothed_kernel_function(Ktyp, reference_point, rb, ha) + Smoothed_kernel_function(Ktyp, reference_point, rb, hs[i]))
        end

        mbWlρb = mb * W/ρb
        mWlρ += mbWlρb
        A += Ab * mbWlρb
    end
    # Shepard normalization
    A /= mWlρ
    return A
end

## Muti-columns intepolation
@inline function _quantities_interpolate_kernel!(output :: O, input::ITPINPUT, reference_point::NTuple{3, T}, ha :: T, neighbors :: NeighborSelection, itp_strategy :: InterpolationStrategy = itpSymmetric) where {ITPINPUT <: AbstractInterpolationInput, T <: AbstractFloat, O<:AbstractVector{T}}
    val_len = Val(length(input.quant))
    columns = ntuple(identity, val_len)
    return _quantities_interpolate_kernel!(output, input, reference_point, ha, neighbors, columns, itp_strategy)
end

@inline function _quantities_interpolate_kernel!(output :: O, input::ITPINPUT, reference_point::NTuple{3, T}, ha :: T, neighbors :: NeighborSelection, columns::NTuple{M,Int}, itp_strategy :: InterpolationStrategy) where {ITPINPUT <: AbstractInterpolationInput, T <: AbstractFloat, O<:AbstractVector{T}, M}
    neighbor_indices = neighbors.pool
    Npart :: Int64 = neighbors.count
    ncols = length(columns)

    @assert length(output) == ncols "Length of `output` must match the requested columns."

    if Npart == 0 || ncols == 0
        fill!(output, T(NaN))
        return
    end

    xs = input.x
    ys = input.y
    zs = input.z
    ms = input.m
    hs = input.h
    ρs = input.ρ
    vals = input.quant
    density_flags = ntuple(j -> columns[j] == 0, ncols)
    has_nondensity = any(!flag for flag in density_flags)
    Ktyp = input.smoothed_kernel

    mWlρ :: T = zero(T)
    fill!(output, zero(T))

    @inbounds for k in 1:Npart
        i = neighbor_indices[k]
        mb = ms[i]
        ρb = ρs[i]
        rb :: NTuple{3, T} = (xs[i], ys[i], zs[i])
        W :: T = zero(T)
        if itp_strategy == itpGather
            W = Smoothed_kernel_function(Ktyp, reference_point, rb, ha)
        elseif itp_strategy == itpScatter
            W = Smoothed_kernel_function(Ktyp, reference_point, rb, hs[i])
        elseif itp_strategy == itpSymmetric
            W = T(0.5) * (Smoothed_kernel_function(Ktyp, reference_point, rb, ha) + Smoothed_kernel_function(Ktyp, reference_point, rb, hs[i]))
        end

        mbW = mb * W
        mbWlρb = mbW/ρb
        if has_nondensity
            mWlρ += mbWlρb
        end

        @inbounds for j in 1:ncols
            if density_flags[j]
                output[j] += mbW
            else
                col_idx = columns[j]
                output[j] += mbWlρb * vals[col_idx][i]
            end
        end
    end

    if has_nondensity
        @inbounds for j in 1:ncols
            density_flags[j] && continue
            output[j] /= mWlρ
        end
    end
    return
end

@inline function _quantities_interpolate_kernel!(output :: O, input::ITPINPUT, reference_point::NTuple{3, T}, ha :: T, neighbors :: NeighborSelection, columns::NTuple{M,Int}) where {ITPINPUT <: AbstractInterpolationInput, T <: AbstractFloat, O<:AbstractVector{T}, M}
    return _quantities_interpolate_kernel!(output, input, reference_point, ha, neighbors, columns, itpSymmetric)
end

## LOS density interpolation (Column / Surface density)
@inline function _LOS_density_kernel(input::ITPINPUT, reference_point::NTuple{2, T}, ha :: T, neighbors :: NeighborSelection, itp_strategy :: InterpolationStrategy = itpSymmetric) where {ITPINPUT <: AbstractInterpolationInput, T <: AbstractFloat}
    # Return 0.0 if no particle in the data
    neighbor_indices = neighbors.pool
    Npart :: Int64 = neighbors.count
    if Npart == 0
        return zero(T)
    end

    # Prepare for interpolation
    # Known quantities
    xs = input.x
    ys = input.y
    ms = input.m
    hs = input.h
    Ktyp = input.smoothed_kernel

    # Initialize counter
    Sigma :: T = zero(T)

    @inbounds for k in 1:Npart
        i = neighbor_indices[k]
        mb = ms[i]
        rb :: NTuple{2, T} = (xs[i], ys[i])
        W :: T = zero(T)
        if itp_strategy == itpGather
            W = LOSint_Smoothed_kernel_function(Ktyp, reference_point, rb, ha)
        elseif itp_strategy == itpScatter
            W = LOSint_Smoothed_kernel_function(Ktyp, reference_point, rb, hs[i])
        elseif itp_strategy == itpSymmetric
            W = T(0.5) * (LOSint_Smoothed_kernel_function(Ktyp, reference_point, rb, ha) + LOSint_Smoothed_kernel_function(Ktyp, reference_point, rb, hs[i]))
        end

        # Counting
        Sigma += mb * W     
    end
    return Sigma
end

## LOS quantities interpolation
@inline function _LOS_quantities_interpolate_kernel!(output :: O, input::ITPINPUT, reference_point::NTuple{2, T}, ha :: T, neighbors :: NeighborSelection, itp_strategy :: InterpolationStrategy = itpSymmetric) where {ITPINPUT <: AbstractInterpolationInput, T <: AbstractFloat, O<:AbstractVector{T}}
    val_len = Val(length(input.quant))
    columns = ntuple(identity, val_len)
    return _LOS_quantities_interpolate_kernel!(output, input, reference_point, ha, neighbors, columns, itp_strategy)
end

@inline function _LOS_quantities_interpolate_kernel!(output :: O, input::ITPINPUT, reference_point::NTuple{2, T}, ha :: T, neighbors :: NeighborSelection, columns::NTuple{M,Int}, itp_strategy :: InterpolationStrategy) where {ITPINPUT <: AbstractInterpolationInput, T <: AbstractFloat, O<:AbstractVector{T}, M}
    neighbor_indices = neighbors.pool
    Npart :: Int64 = neighbors.count
    ncols = length(columns)

    @assert length(output) == ncols "Length of `output` must match the requested columns."

    if Npart == 0 || ncols == 0
        fill!(output, T(NaN))
        return
    end

    xs = input.x
    ys = input.y
    ms = input.m
    hs = input.h
    ρs = input.ρ
    vals = input.quant
    density_flags = ntuple(j -> columns[j] == 0, ncols)
    has_nondensity = any(!flag for flag in density_flags)
    Ktyp = input.smoothed_kernel

    mWlρ :: T = zero(T)
    fill!(output, zero(T))

    @inbounds for k in 1:Npart
        i = neighbor_indices[k]
        mb = ms[i]
        ρb = ρs[i]
        rb :: NTuple{2, T} = (xs[i], ys[i])
        W :: T = zero(T)
        if itp_strategy == itpGather
            W = LOSint_Smoothed_kernel_function(Ktyp, reference_point, rb, ha)
        elseif itp_strategy == itpScatter
            W = LOSint_Smoothed_kernel_function(Ktyp, reference_point, rb, hs[i])
        elseif itp_strategy == itpSymmetric
            W = T(0.5) * (LOSint_Smoothed_kernel_function(Ktyp, reference_point, rb, ha) + LOSint_Smoothed_kernel_function(Ktyp, reference_point, rb, hs[i]))
        end

        mbW = mb * W
        mbWlρb = mbW/ρb
        if has_nondensity
            mWlρ += mbWlρb
        end

        @inbounds for j in 1:ncols
            if density_flags[j]
                output[j] += mbW
            else
                col_idx = columns[j]
                output[j] += mbWlρb * vals[col_idx][i]
            end
        end
    end

    if has_nondensity
        @inbounds for j in 1:ncols
            density_flags[j] && continue
            output[j] /= mWlρ
        end
    end
    return
end

@inline function _LOS_quantities_interpolate_kernel!(output :: O, input::ITPINPUT, reference_point::NTuple{2, T}, ha :: T, neighbors :: NeighborSelection, columns::NTuple{M,Int}) where {ITPINPUT <: AbstractInterpolationInput, T <: AbstractFloat, O<:AbstractVector{T}, M}
    return _LOS_quantities_interpolate_kernel!(output, input, reference_point, ha, neighbors, columns, itpSymmetric)
end

"""
∇ρ(r) = (1/ρ(r))∑_b m_b*(ρ_b-ρ(r))∇W(r-r_b)
      = (1/ρ(r))((∑_b m_b*ρ_b*∇W(r-r_b))  - ρ(r)(∑_b m_b*∇W(r-r_b))
      = (1/ρ(r))((∑_b m_b*ρ_b*∇W(r-r_b)) - ∑_b m_b*∇W(r-r_b)
"""
# Single column gradient density intepolation
@inline function _gradient_density_kernel(input::ITPINPUT, reference_point::NTuple{3, T}, ha :: T, neighbors :: NeighborSelection, itp_strategy :: InterpolationStrategy = itpSymmetric) where {ITPINPUT <: AbstractInterpolationInput, T <: AbstractFloat}
    # Return (NaN) if no particle in the data
    neighbor_indices = neighbors.pool
    Npart :: Int64 = neighbors.count
    if Npart == 0
        return (T(NaN), T(NaN), T(NaN))
    end

    # Prepare for interpolation
    # Known quantities
    xs = input.x
    ys = input.y
    zs = input.z
    ms = input.m
    hs = input.h
    ρs = input.ρ
    Ktyp = input.smoothed_kernel

    # Initialize counter
    ∇ρxf :: T = zero(T)
    ∇ρyf :: T = zero(T)
    ∇ρzf :: T = zero(T)
    ∇ρxb :: T = zero(T)
    ∇ρyb :: T = zero(T)
    ∇ρzb :: T = zero(T)

    ρ :: T = zero(T)

    @inbounds for k in 1:Npart
        i = neighbor_indices[k]
        mb = ms[i]
        ρb = ρs[i]
        rb :: NTuple{3, T} = (xs[i], ys[i], zs[i])
        W :: T = zero(T)
        if itp_strategy == itpGather
            W = Smoothed_kernel_function(Ktyp, reference_point, rb, ha)
        elseif itp_strategy == itpScatter
            W = Smoothed_kernel_function(Ktyp, reference_point, rb, hs[i])
        elseif itp_strategy == itpSymmetric
            W = T(0.5) * (Smoothed_kernel_function(Ktyp, reference_point, rb, ha) + Smoothed_kernel_function(Ktyp, reference_point, rb, hs[i]))
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
            ∇W = Smoothed_gradient_kernel_function(Ktyp, reference_point, rb, hs[i])
            ∂xW = ∇W[1]
            ∂yW = ∇W[2]
            ∂zW = ∇W[3]
        elseif itp_strategy == itpSymmetric
            ∇Wa = Smoothed_gradient_kernel_function(Ktyp, reference_point, rb, ha)
            ∇Wb = Smoothed_gradient_kernel_function(Ktyp, reference_point, rb, hs[i])
            ∂xW = T(0.5) * (∇Wa[1] + ∇Wb[1])
            ∂yW = T(0.5) * (∇Wa[2] + ∇Wb[2])
            ∂zW = T(0.5) * (∇Wa[3] + ∇Wb[3])
        end

        # Counting
        ## Normal
        mbW = mb * W
        ρ += mbW                                # ρ(r)

        # Gradient
        mb∂xW = mb * ∂xW
        mb∂yW = mb * ∂yW
        mb∂zW = mb * ∂zW

        ∇ρxf += mb∂xW * ρb
        ∇ρyf += mb∂yW * ρb
        ∇ρzf += mb∂zW * ρb
        ∇ρxb += mb∂xW
        ∇ρyb += mb∂yW
        ∇ρzb += mb∂zW
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
@inline function _gradient_quantity_interpolate_kernel(input::ITPINPUT, reference_point::NTuple{3, T}, ha :: T, neighbors :: NeighborSelection, column_idx :: Int64, itp_strategy :: InterpolationStrategy = itpSymmetric) where {ITPINPUT <: AbstractInterpolationInput, T <: AbstractFloat}
    # Return (NaN) if no particle in the data
    neighbor_indices = neighbors.pool
    Npart :: Int64 = neighbors.count
    if Npart == 0
        return (T(NaN), T(NaN), T(NaN))
    end

    # Prepare for interpolation
    # Known quantities
    xs = input.x
    ys = input.y
    zs = input.z
    ms = input.m
    hs = input.h
    ρs = input.ρ
    As = input.quant[column_idx]
    Ktyp = input.smoothed_kernel

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

    @inbounds for k in 1:Npart
        i = neighbor_indices[k]
        mb = ms[i]
        ρb = ρs[i]
        Ab = As[i]
        rb :: NTuple{3, T} = (xs[i], ys[i], zs[i])
        W :: T = zero(T)
        if itp_strategy == itpGather
            W = Smoothed_kernel_function(Ktyp, reference_point, rb, ha)
        elseif itp_strategy == itpScatter
            W = Smoothed_kernel_function(Ktyp, reference_point, rb, hs[i])
        elseif itp_strategy == itpSymmetric
            W = T(0.5) * (Smoothed_kernel_function(Ktyp, reference_point, rb, ha) + Smoothed_kernel_function(Ktyp, reference_point, rb, hs[i]))
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
            ∇W = Smoothed_gradient_kernel_function(Ktyp, reference_point, rb, hs[i])
            ∂xW = ∇W[1]
            ∂yW = ∇W[2]
            ∂zW = ∇W[3]
        elseif itp_strategy == itpSymmetric
            ∇Wa = Smoothed_gradient_kernel_function(Ktyp, reference_point, rb, ha)
            ∇Wb = Smoothed_gradient_kernel_function(Ktyp, reference_point, rb, hs[i])
            ∂xW = T(0.5) * (∇Wa[1] + ∇Wb[1])
            ∂yW = T(0.5) * (∇Wa[2] + ∇Wb[2])
            ∂zW = T(0.5) * (∇Wa[3] + ∇Wb[3])
        end

        # Counting
        ## Normal
        mbW = mb * W
        mWlρ += mbW/ρb                          # Shepard normalization for A(r)
        ρ += mbW                                # ρ(r)
        A += (mbW * Ab)/ρb                      # A(r)

        # Gradient
        mb∂xW = mb * ∂xW
        mb∂yW = mb * ∂yW
        mb∂zW = mb * ∂zW

        ∇Axf += mb∂xW * Ab
        ∇Ayf += mb∂yW * Ab
        ∇Azf += mb∂zW * Ab
        ∇Axb += mb∂xW
        ∇Ayb += mb∂yW
        ∇Azb += mb∂zW
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
@inline function _divergence_quantity_interpolate_kernel(input::ITPINPUT, reference_point::NTuple{3, T}, ha :: T, neighbors :: NeighborSelection, Ax_column_idx :: Int64, Ay_column_idx :: Int64, Az_column_idx :: Int64, itp_strategy :: InterpolationStrategy = itpSymmetric) where {ITPINPUT <: AbstractInterpolationInput, T <: AbstractFloat}
    # Return 0.0 if no particle in the data
    neighbor_indices = neighbors.pool
    Npart :: Int64 = neighbors.count
    if Npart == 0
        return T(NaN)
    end

    # Prepare for interpolation
    # Known quantities
    xs = input.x
    ys = input.y
    zs = input.z
    ms = input.m
    hs = input.h
    ρs = input.ρ
    Axs = input.quant[Ax_column_idx]
    Ays = input.quant[Ay_column_idx]
    Azs = input.quant[Az_column_idx]
    Ktyp = input.smoothed_kernel

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
    @inbounds for k in 1:Npart
        i = neighbor_indices[k]
        mb = ms[i]
        ρb = ρs[i]
        Axb = Axs[i]
        Ayb = Ays[i]
        Azb = Azs[i]
        rb :: NTuple{3, T} = (xs[i], ys[i], zs[i])
        W :: T = zero(T)
        if itp_strategy == itpGather
            W = Smoothed_kernel_function(Ktyp, reference_point, rb, ha)
        elseif itp_strategy == itpScatter
            W = Smoothed_kernel_function(Ktyp, reference_point, rb, hs[i])
        elseif itp_strategy == itpSymmetric
            W = T(0.5) * (Smoothed_kernel_function(Ktyp, reference_point, rb, ha) + Smoothed_kernel_function(Ktyp, reference_point, rb, hs[i]))
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
            ∇W = Smoothed_gradient_kernel_function(Ktyp, reference_point, rb, hs[i])
            ∂xW = ∇W[1]
            ∂yW = ∇W[2]
            ∂zW = ∇W[3]
        elseif itp_strategy == itpSymmetric
            ∇Wa = Smoothed_gradient_kernel_function(Ktyp, reference_point, rb, ha)
            ∇Wb = Smoothed_gradient_kernel_function(Ktyp, reference_point, rb, hs[i])
            ∂xW = T(0.5) * (∇Wa[1] + ∇Wb[1])
            ∂yW = T(0.5) * (∇Wa[2] + ∇Wb[2])
            ∂zW = T(0.5) * (∇Wa[3] + ∇Wb[3])
        end

        # Counting
        ## Normal
        mbW = mb * W
        mWlρ += mbW/ρb                          # Shepard normalization for A(r)
        ρ += mbW                                # ρ(r)
        Ax += (mbW * Axb)/ρb                    # Ax(r)
        Ay += (mbW * Ayb)/ρb                    # Ay(r)
        Az += (mbW * Azb)/ρb                    # Az(r)

        # Gradient
        mb∂xW = mb * ∂xW
        mb∂yW = mb * ∂yW
        mb∂zW = mb * ∂zW

        ∇Af += mb∂xW * Axb + mb∂yW * Ayb + mb∂zW * Azb
        ∇Axb += mb∂xW
        ∇Ayb += mb∂yW
        ∇Azb += mb∂zW
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
@inline function _curl_quantity_interpolate_kernel(input::ITPINPUT, reference_point::NTuple{3, T}, ha :: T, neighbors :: NeighborSelection, Ax_column_idx :: Int64, Ay_column_idx :: Int64, Az_column_idx :: Int64, itp_strategy :: InterpolationStrategy = itpSymmetric) where {ITPINPUT <: AbstractInterpolationInput, T <: AbstractFloat}
    # Return 0.0 if no particle in the data
    neighbor_indices = neighbors.pool
    Npart :: Int64 = neighbors.count
    if Npart == 0
        return (T(NaN), T(NaN), T(NaN))
    end

    # Prepare for interpolation
    # Known quantities
    xs = input.x
    ys = input.y
    zs = input.z
    ms = input.m
    hs = input.h
    ρs = input.ρ
    Axs = input.quant[Ax_column_idx]
    Ays = input.quant[Ay_column_idx]
    Azs = input.quant[Az_column_idx]
    Ktyp = input.smoothed_kernel

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
    @inbounds for k in 1:Npart
        i = neighbor_indices[k]
        mb = ms[i]
        ρb = ρs[i]
        Axb = Axs[i]
        Ayb = Ays[i]
        Azb = Azs[i]
        rb :: NTuple{3, T} = (xs[i], ys[i], zs[i])

        W :: T = zero(T)
        if itp_strategy == itpGather
            W = Smoothed_kernel_function(Ktyp, reference_point, rb, ha)
        elseif itp_strategy == itpScatter
            W = Smoothed_kernel_function(Ktyp, reference_point, rb, hs[i])
        elseif itp_strategy == itpSymmetric
            W = T(0.5) * (Smoothed_kernel_function(Ktyp, reference_point, rb, ha) + Smoothed_kernel_function(Ktyp, reference_point, rb, hs[i]))
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
            ∇W = Smoothed_gradient_kernel_function(Ktyp, reference_point, rb, hs[i])
            ∂xW = ∇W[1]
            ∂yW = ∇W[2]
            ∂zW = ∇W[3]
        elseif itp_strategy == itpSymmetric
            ∇Wa = Smoothed_gradient_kernel_function(Ktyp, reference_point, rb, ha)
            ∇Wb = Smoothed_gradient_kernel_function(Ktyp, reference_point, rb, hs[i])
            ∂xW = T(0.5) * (∇Wa[1] + ∇Wb[1])
            ∂yW = T(0.5) * (∇Wa[2] + ∇Wb[2])
            ∂zW = T(0.5) * (∇Wa[3] + ∇Wb[3])
        end

        # Counting
        ## Normal
        mbW = mb * W
        mWlρ += mbW/ρb                          # Shepard normalization for A(r)
        ρ += mbW                                # ρ(r)
        Ax += (mbW * Axb)/ρb                    # Ax(r)
        Ay += (mbW * Ayb)/ρb                    # Ay(r)
        Az += (mbW * Azb)/ρb                    # Az(r)

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