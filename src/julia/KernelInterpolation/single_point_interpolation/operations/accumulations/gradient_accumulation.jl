"""
∇ρ(r) = ∑_b m_b/ρ_b*(ρ_b-ρ(r))∇W(r-r_b)
      = ∑_b m_b*∇W(r-r_b)  - ρ(r)(∑_b m_b/ρ_b*∇W(r-r_b)
      = ∇ρf - ∇ρb
"""
# Single column gradient density intepolation
@inline function _gradient_density_accumulation(ra::NTuple{3, T}, rb::NTuple{3, T}, mb :: T, ρb :: T, h :: T, smoothed_kernel :: K) where {T <: AbstractFloat, K <: AbstractSPHKernel}
    Ktyp = typeof(smoothed_kernel)
    ∇W = Smoothed_gradient_kernel_function(Ktyp, ra, rb, h)
    ∂xW = ∇W[1]
    ∂yW = ∇W[2]
    ∂zW = ∇W[3]

    invρb = inv(ρb)

    # Gradient
    mb∂xW = mb * ∂xW
    mb∂yW = mb * ∂yW
    mb∂zW = mb * ∂zW

    ∇ρxf = mb∂xW
    ∇ρyf = mb∂yW
    ∇ρzf = mb∂zW
    ∇ρxb = mb∂xW * invρb
    ∇ρyb = mb∂yW * invρb
    ∇ρzb = mb∂zW * invρb
    return ∇ρxf, ∇ρyf, ∇ρzf, ∇ρxb, ∇ρyb, ∇ρzb
end

@inline function _gradient_density_accumulation(ra::NTuple{3, T}, rb::NTuple{3, T}, mb :: T, ρb :: T, ha :: T, hb :: T, smoothed_kernel :: K) where {T <: AbstractFloat, K <: AbstractSPHKernel}
    Ktyp = typeof(smoothed_kernel)
    ∇Wa = Smoothed_gradient_kernel_function(Ktyp, ra, rb, ha)
    ∇Wb = Smoothed_gradient_kernel_function(Ktyp, ra, rb, hb)
    ∂xW = T(0.5) * (∇Wa[1] + ∇Wb[1])
    ∂yW = T(0.5) * (∇Wa[2] + ∇Wb[2])
    ∂zW = T(0.5) * (∇Wa[3] + ∇Wb[3])

    invρb = inv(ρb)

    # Gradient
    mb∂xW = mb * ∂xW
    mb∂yW = mb * ∂yW
    mb∂zW = mb * ∂zW

    ∇ρxf = mb∂xW
    ∇ρyf = mb∂yW
    ∇ρzf = mb∂zW
    ∇ρxb = mb∂xW * invρb
    ∇ρyb = mb∂yW * invρb
    ∇ρzb = mb∂zW * invρb
    return ∇ρxf, ∇ρyf, ∇ρzf, ∇ρxb, ∇ρyb, ∇ρzb
end

"""
∇A(r) = ∑_b m_b/ρ_b*(A_b-A(r))∇W(r-r_b)
      = ∑_b m_b/ρ_b*A_b*∇W(r-r_b))  - A(r)(∑_b m_b/ρ_b*∇W(r-r_b))
      = ∇Af - ∇Ab
"""
# Single column gradient value intepolation
@inline function _gradient_quantity_accumulation(ra::NTuple{3, T}, rb::NTuple{3, T}, mb :: T, ρb :: T, Ab :: T, h :: T, smoothed_kernel :: K) where {T <: AbstractFloat, K <: AbstractSPHKernel}
    Ktyp = typeof(smoothed_kernel)
    ∇W = Smoothed_gradient_kernel_function(Ktyp, ra, rb, h)
    ∂xW = ∇W[1]
    ∂yW = ∇W[2]
    ∂zW = ∇W[3]

    invρb = inv(ρb)

    # Gradient
    mblρb∂xW = mb * invρb * ∂xW
    mblρb∂yW = mb * invρb * ∂yW
    mblρb∂zW = mb * invρb * ∂zW

    ∇Axf = mblρb∂xW * Ab
    ∇Ayf = mblρb∂yW * Ab
    ∇Azf = mblρb∂zW * Ab
    ∇Axb = mblρb∂xW
    ∇Ayb = mblρb∂yW
    ∇Azb = mblρb∂zW
    return ∇Axf, ∇Ayf, ∇Azf, ∇Axb, ∇Ayb, ∇Azb
end

@inline function _gradient_quantity_accumulation(ra::NTuple{3, T}, rb::NTuple{3, T}, mb :: T, ρb :: T, Ab :: T, ha :: T, hb :: T, smoothed_kernel :: K) where {T <: AbstractFloat, K <: AbstractSPHKernel}
    Ktyp = typeof(smoothed_kernel)
    ∇Wa = Smoothed_gradient_kernel_function(Ktyp, ra, rb, ha)
    ∇Wb = Smoothed_gradient_kernel_function(Ktyp, ra, rb, hb)
    ∂xW = T(0.5) * (∇Wa[1] + ∇Wb[1])
    ∂yW = T(0.5) * (∇Wa[2] + ∇Wb[2])
    ∂zW = T(0.5) * (∇Wa[3] + ∇Wb[3])

    invρb = inv(ρb)

    # Gradient
    mblρb∂xW = mb * invρb * ∂xW
    mblρb∂yW = mb * invρb * ∂yW
    mblρb∂zW = mb * invρb * ∂zW

    ∇Axf = mblρb∂xW * Ab
    ∇Ayf = mblρb∂yW * Ab
    ∇Azf = mblρb∂zW * Ab
    ∇Axb = mblρb∂xW
    ∇Ayb = mblρb∂yW
    ∇Azb = mblρb∂zW
    return ∇Axf, ∇Ayf, ∇Azf, ∇Axb, ∇Ayb, ∇Azb
end