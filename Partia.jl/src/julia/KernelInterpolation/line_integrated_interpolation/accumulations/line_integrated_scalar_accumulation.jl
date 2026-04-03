## Line-integrated density interpolation (Column / Surface density)
@inline function _line_integrated_density_accumulation(Δr :: T, mb :: T, h :: T, smoothed_kernel :: K) where {T <: AbstractFloat, K <: AbstractSPHKernel}
    Ktyp = typeof(smoothed_kernel)
    W = line_integrated_kernel_function(Ktyp, Δr, h)
    return mb * W
end

@inline function _line_integrated_density_accumulation(Δr :: T, mb :: T, ha :: T, hb :: T, smoothed_kernel :: K) where {T <: AbstractFloat, K <: AbstractSPHKernel}
    Ktyp = typeof(smoothed_kernel)
    W = T(0.5) * (line_integrated_kernel_function(Ktyp, Δr, ha) + line_integrated_kernel_function(Ktyp, Δr, hb))
    return mb * W
end

## Line-integrated quantities interpolation
@inline function _line_integrated_quantity_interpolate_accumulation(Δr :: T, mb :: T, ρb :: T, Ab :: T, h :: T, smoothed_kernel :: K) where {T <: AbstractFloat, K <: AbstractSPHKernel}
    Ktyp = typeof(smoothed_kernel)
    W = line_integrated_kernel_function(Ktyp, Δr, h)
    mbWlρ = mb * W / ρb
    return Ab * mbWlρ
end

@inline function _line_integrated_quantity_interpolate_accumulation(Δr :: T, mb :: T, ρb :: T, Ab :: T, ha :: T, hb :: T, smoothed_kernel :: K) where {T <: AbstractFloat, K <: AbstractSPHKernel}
    Ktyp = typeof(smoothed_kernel)
    W = T(0.5) * (line_integrated_kernel_function(Ktyp, Δr, ha) + line_integrated_kernel_function(Ktyp, Δr, hb))
    mbWlρ = mb * W / ρb
    return Ab * mbWlρ
end

## Line-integrated Shepard normalization
@inline function _line_integrated_ShepardNormalization_accumulation(Δr :: T, mb :: T, ρb :: T, h :: T, smoothed_kernel :: K) where {T <: AbstractFloat, K <: AbstractSPHKernel}
    Ktyp = typeof(smoothed_kernel)
    W = line_integrated_kernel_function(Ktyp, Δr, h)
    mbWlρ = mb * W / ρb
    return mbWlρ
end

@inline function _line_integrated_ShepardNormalization_accumulation(Δr :: T, mb :: T, ρb :: T, ha :: T, hb :: T, smoothed_kernel :: K) where {T <: AbstractFloat, K <: AbstractSPHKernel}
    Ktyp = typeof(smoothed_kernel)
    W = T(0.5) * (line_integrated_kernel_function(Ktyp, Δr, ha) + line_integrated_kernel_function(Ktyp, Δr, hb))
    mbWlρ = mb * W / ρb
    return mbWlρ
end
