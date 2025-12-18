"""
∇×A(r) = -(1/ρ(r))∑_b m_b*(A_b-A(r))×∇W(r-r_b)
       = -(1/ρ(r)) * ((∑_b m_b*A_b×∇W(r-r_b)) - A(r)×(∑_b m_b*∇W(r-r_b)))
       = -(1/ρ(r))*(∇×Af - ∇×Ab)
"""
# Single column curl value intepolation
@inline function _curl_quantity_accumulation(input :: ITPINPUT, reference_point::NTuple{3, T}, ha :: T, Ax_column_idx :: Int, Ay_column_idx :: Int, Az_column_idx :: Int, :: Type{itpGather}, i :: Int) where {ITPINPUT <: AbstractInterpolationInput, T <: AbstractFloat}
    Ktyp = typeof(input.smoothed_kernel)
    mb = input.m[i]
    Axb = input.quant[Ax_column_idx][i]
    Ayb = input.quant[Ay_column_idx][i]
    Azb = input.quant[Az_column_idx][i]
    rb :: NTuple{3, T} = (input.x[i], input.y[i], input.z[i])
    ∇W = Smoothed_gradient_kernel_function(Ktyp, reference_point, rb, ha)
    ∂xW = ∇W[1]
    ∂yW = ∇W[2]
    ∂zW = ∇W[3]

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

@inline function _curl_quantity_accumulation(input :: ITPINPUT, reference_point::NTuple{3, T}, ha :: T, Ax_column_idx :: Int, Ay_column_idx :: Int, Az_column_idx :: Int, :: Type{itpScatter}, i :: Int) where {ITPINPUT <: AbstractInterpolationInput, T <: AbstractFloat}
    Ktyp = typeof(input.smoothed_kernel)
    mb = input.m[i]
    Axb = input.quant[Ax_column_idx][i]
    Ayb = input.quant[Ay_column_idx][i]
    Azb = input.quant[Az_column_idx][i]
    rb :: NTuple{3, T} = (input.x[i], input.y[i], input.z[i])
    ∇W = Smoothed_gradient_kernel_function(Ktyp, reference_point, rb, input.h[i])
    ∂xW = ∇W[1]
    ∂yW = ∇W[2]
    ∂zW = ∇W[3]

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

@inline function _curl_quantity_accumulation(input :: ITPINPUT, reference_point::NTuple{3, T}, ha :: T, Ax_column_idx :: Int, Ay_column_idx :: Int, Az_column_idx :: Int, :: Type{itpSymmetric}, i :: Int) where {ITPINPUT <: AbstractInterpolationInput, T <: AbstractFloat}
    Ktyp = typeof(input.smoothed_kernel)
    mb = input.m[i]
    Axb = input.quant[Ax_column_idx][i]
    Ayb = input.quant[Ay_column_idx][i]
    Azb = input.quant[Az_column_idx][i]
    rb :: NTuple{3, T} = (input.x[i], input.y[i], input.z[i])
    ∇Wa = Smoothed_gradient_kernel_function(Ktyp, reference_point, rb, ha)
    ∇Wb = Smoothed_gradient_kernel_function(Ktyp, reference_point, rb, input.h[i])
    ∂xW = T(0.5) * (∇Wa[1] + ∇Wb[1])
    ∂yW = T(0.5) * (∇Wa[2] + ∇Wb[2])
    ∂zW = T(0.5) * (∇Wa[3] + ∇Wb[3])

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