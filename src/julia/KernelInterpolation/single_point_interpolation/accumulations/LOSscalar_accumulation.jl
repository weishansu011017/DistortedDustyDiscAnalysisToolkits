## LOS density interpolation (Column / Surface density)
@inline function _LOS_density_accumulation(ra::NTuple{2, T}, rb::NTuple{2, T}, mb :: T, h :: T, smoothed_kernel :: K) where {T <: AbstractFloat, K <: AbstractSPHKernel}
    Ktyp = typeof(smoothed_kernel)
    W = LOSint_Smoothed_kernel_function(Ktyp, ra, rb, h)
    return mb * W
end

@inline function _LOS_density_accumulation(ra::NTuple{2, T}, rb::NTuple{2, T}, mb :: T, ha :: T, hb :: T, smoothed_kernel :: K) where {T <: AbstractFloat, K <: AbstractSPHKernel}
    Ktyp = typeof(smoothed_kernel)
    W = T(0.5) * (LOSint_Smoothed_kernel_function(Ktyp, ra, rb, ha) + LOSint_Smoothed_kernel_function(Ktyp, ra, rb, hb))
    return mb * W
end

## LOS quantities interpolation
@inline function _LOS_quantity_interpolate_accumulation(ra::NTuple{2, T}, rb::NTuple{2, T}, mb :: T, ρb :: T, Ab :: T, h :: T, smoothed_kernel :: K) where {T <: AbstractFloat, K <: AbstractSPHKernel}
    Ktyp = typeof(smoothed_kernel)
    W = LOSint_Smoothed_kernel_function(Ktyp, ra, rb, h)
    mbWlρb = mb * W/ρb
    return Ab * mbWlρb
end

@inline function _LOS_quantity_interpolate_accumulation(ra::NTuple{2, T}, rb::NTuple{2, T}, mb :: T, ρb :: T, Ab :: T, ha :: T, hb :: T, smoothed_kernel :: K) where {T <: AbstractFloat, K <: AbstractSPHKernel}
    Ktyp = typeof(smoothed_kernel)
    W = T(0.5) * (LOSint_Smoothed_kernel_function(Ktyp, ra, rb, ha) + LOSint_Smoothed_kernel_function(Ktyp, ra, rb, hb))
    mbWlρb = mb * W/ρb
    return Ab * mbWlρb
end

@inline function _LOS_quantity_interpolate_accumulation(ra::NTuple{3, T}, rb::NTuple{3, T}, mb :: T, ρb :: T, Ab :: T, h :: T, smoothed_kernel :: K) where {T <: AbstractFloat, K <: AbstractSPHKernel}
    ra2d :: NTuple{2, T} = (ra[1], ra[2])
    rb2d :: NTuple{2, T} = (rb[1], rb[2])
    Ktyp = typeof(smoothed_kernel)
    W = LOSint_Smoothed_kernel_function(Ktyp, ra2d, rb2d, h)
    mbWlρb = mb * W/ρb
    return Ab * mbWlρb
end

@inline function _LOS_quantity_interpolate_accumulation(ra::NTuple{3, T}, rb::NTuple{3, T}, mb :: T, ρb :: T, Ab :: T, ha :: T, hb :: T, smoothed_kernel :: K) where {T <: AbstractFloat, K <: AbstractSPHKernel}
    ra2d :: NTuple{2, T} = (ra[1], ra[2])
    rb2d :: NTuple{2, T} = (rb[1], rb[2])
    Ktyp = typeof(smoothed_kernel)
    W = T(0.5) * (LOSint_Smoothed_kernel_function(Ktyp, ra2d, rb2d, ha) + LOSint_Smoothed_kernel_function(Ktyp, ra2d, rb2d, hb))
    mbWlρb = mb * W/ρb
    return Ab * mbWlρb
end

@inline function _LOS_ShepardNormalization_accumulation(ra::NTuple{2, T}, rb::NTuple{2, T}, mb :: T, ρb :: T, h :: T, smoothed_kernel :: K) where {T <: AbstractFloat, K <: AbstractSPHKernel}
    Ktyp = typeof(smoothed_kernel)
    W = LOSint_Smoothed_kernel_function(Ktyp, ra, rb, h)
    mbWlρb = mb * W/ρb
    return mbWlρb
end

@inline function _LOS_ShepardNormalization_accumulation(ra::NTuple{2, T}, rb::NTuple{2, T}, mb :: T, ρb :: T, ha :: T, hb :: T, smoothed_kernel :: K) where {T <: AbstractFloat, K <: AbstractSPHKernel}
    Ktyp = typeof(smoothed_kernel)
    W = T(0.5) * (LOSint_Smoothed_kernel_function(Ktyp, ra, rb, ha) + LOSint_Smoothed_kernel_function(Ktyp, ra, rb, hb))
    mbWlρb = mb * W/ρb
    return mbWlρb
end

@inline function _LOS_ShepardNormalization_accumulation(ra::NTuple{3, T}, rb::NTuple{3, T}, mb :: T, ρb :: T, h :: T, smoothed_kernel :: K) where {T <: AbstractFloat, K <: AbstractSPHKernel}
    ra2d :: NTuple{2, T} = (ra[1], ra[2])
    rb2d :: NTuple{2, T} = (rb[1], rb[2])
    Ktyp = typeof(smoothed_kernel)
    W = LOSint_Smoothed_kernel_function(Ktyp, ra2d, rb2d, h)
    mbWlρb = mb * W/ρb
    return mbWlρb
end

@inline function _LOS_ShepardNormalization_accumulation(ra::NTuple{3, T}, rb::NTuple{3, T}, mb :: T, ρb :: T, ha :: T, hb :: T, smoothed_kernel :: K) where {T <: AbstractFloat, K <: AbstractSPHKernel}
    ra2d :: NTuple{2, T} = (ra[1], ra[2])
    rb2d :: NTuple{2, T} = (rb[1], rb[2])
    Ktyp = typeof(smoothed_kernel)
    W = T(0.5) * (LOSint_Smoothed_kernel_function(Ktyp, ra2d, rb2d, ha) + LOSint_Smoothed_kernel_function(Ktyp, ra2d, rb2d, hb))
    mbWlρb = mb * W/ρb
    return mbWlρb
end