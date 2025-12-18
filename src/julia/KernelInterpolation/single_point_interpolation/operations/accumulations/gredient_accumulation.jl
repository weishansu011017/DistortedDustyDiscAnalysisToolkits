
"""
∇ρ(r) = (1/ρ(r))∑_b m_b*(ρ_b-ρ(r))∇W(r-r_b)
      = (1/ρ(r))((∑_b m_b*ρ_b*∇W(r-r_b))  - ρ(r)(∑_b m_b*∇W(r-r_b))
      = (1/ρ(r))((∑_b m_b*ρ_b*∇W(r-r_b)) - ∑_b m_b*∇W(r-r_b)
"""
# Single column gradient density intepolation
@inline function _gradient_density_accumulation(input :: ITPINPUT, reference_point::NTuple{3, T}, ha :: T, :: Type{itpGather}, i :: Int) where {ITPINPUT <: AbstractInterpolationInput, T <: AbstractFloat}
    Ktyp = typeof(input.smoothed_kernel)
    mb = input.m[i]
    ρb = input.ρ[i]
    rb :: NTuple{3, T} = (input.x[i], input.y[i], input.z[i])
    ∇W = Smoothed_gradient_kernel_function(Ktyp, reference_point, rb, ha)
    ∂xW = ∇W[1]
    ∂yW = ∇W[2]
    ∂zW = ∇W[3]

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

@inline function _gradient_density_accumulation(input :: ITPINPUT, reference_point::NTuple{3, T}, ha :: T, :: Type{itpScatter}, i :: Int) where {ITPINPUT <: AbstractInterpolationInput, T <: AbstractFloat}
    Ktyp = typeof(input.smoothed_kernel)
    mb = input.m[i]
    ρb = input.ρ[i]
    rb :: NTuple{3, T} = (input.x[i], input.y[i], input.z[i])
    ∇W = Smoothed_gradient_kernel_function(Ktyp, reference_point, rb, input.h[i])
    ∂xW = ∇W[1]
    ∂yW = ∇W[2]
    ∂zW = ∇W[3]

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

@inline function _gradient_density_accumulation(input :: ITPINPUT, reference_point::NTuple{3, T}, ha :: T, :: Type{itpSymmetric}, i :: Int) where {ITPINPUT <: AbstractInterpolationInput, T <: AbstractFloat}
    Ktyp = typeof(input.smoothed_kernel)
    mb = input.m[i]
    ρb = input.ρ[i]
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

    ∇ρxf = mb∂xW * ρb
    ∇ρyf = mb∂yW * ρb
    ∇ρzf = mb∂zW * ρb
    ∇ρxb = mb∂xW
    ∇ρyb = mb∂yW
    ∇ρzb = mb∂zW
    return ∇ρxf, ∇ρyf, ∇ρzf, ∇ρxb, ∇ρyb, ∇ρzb
end


"""
∇A(r) = (1/ρ(r))∑_b m_b*(A_b-A(r))∇W(r-r_b)
      = (1/ρ(r))((∑_b m_b*A_b*∇W(r-r_b))  - A(r)(∑_b m_b*∇W(r-r_b))
      = ∇Af - ∇Ab
"""
# Single column gradient value intepolation
@inline function _gradient_quantity_accumulation(input :: ITPINPUT, reference_point::NTuple{3, T}, ha :: T, column_idx :: Int64, :: Type{itpGather}, i :: Int) where {ITPINPUT <: AbstractInterpolationInput, T <: AbstractFloat}
    Ktyp = typeof(input.smoothed_kernel)
    mb = input.m[i]
    Ab = input.quant[column_idx][i]
    rb :: NTuple{3, T} = (input.x[i], input.y[i], input.z[i])
    ∇W = Smoothed_gradient_kernel_function(Ktyp, reference_point, rb, ha)
    ∂xW = ∇W[1]
    ∂yW = ∇W[2]
    ∂zW = ∇W[3]

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

@inline function _gradient_quantity_accumulation(input :: ITPINPUT, reference_point::NTuple{3, T}, ha :: T, column_idx :: Int64, :: Type{itpScatter}, i :: Int) where {ITPINPUT <: AbstractInterpolationInput, T <: AbstractFloat}
    Ktyp = typeof(input.smoothed_kernel)
    mb = input.m[i]
    Ab = input.quant[column_idx][i]
    rb :: NTuple{3, T} = (input.x[i], input.y[i], input.z[i])
    ∇W = Smoothed_gradient_kernel_function(Ktyp, reference_point, rb, input.h[i])
    ∂xW = ∇W[1]
    ∂yW = ∇W[2]
    ∂zW = ∇W[3]
    
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

@inline function _gradient_quantity_accumulation(input :: ITPINPUT, reference_point::NTuple{3, T}, ha :: T, column_idx :: Int64, :: Type{itpSymmetric}, i :: Int) where {ITPINPUT <: AbstractInterpolationInput, T <: AbstractFloat}
    Ktyp = typeof(input.smoothed_kernel)
    mb = input.m[i]
    Ab = input.quant[column_idx][i]
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

    ∇Axf = mb∂xW * Ab
    ∇Ayf = mb∂yW * Ab
    ∇Azf = mb∂zW * Ab
    ∇Axb = mb∂xW
    ∇Ayb = mb∂yW
    ∇Azb = mb∂zW
    return ∇Axf, ∇Ayf, ∇Azf, ∇Axb, ∇Ayb, ∇Azb
end