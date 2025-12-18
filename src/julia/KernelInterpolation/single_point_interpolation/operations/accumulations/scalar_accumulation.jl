## Density
@inline function _density_accumulation(ra::NTuple{3, T}, rb::NTuple{3, T}, mb :: T, h :: T, smoothed_kernel :: Type{K}) where {T <: AbstractFloat}
    Ktyp = typeof(smoothed_kernel)
    W = Smoothed_kernel_function(Ktyp, ra, rb, h)
    return mb * W
end

@inline function _density_accumulation(ra::NTuple{3, T}, rb::NTuple{3, T}, mb :: T, ha :: T, hb :: T, smoothed_kernel :: Type{K}) where {T <: AbstractFloat}
    Ktyp = typeof(smoothed_kernel)
    W = T(0.5) * (Smoothed_kernel_function(Ktyp, ra, rb, ha) + Smoothed_kernel_function(Ktyp, ra, rb, hb))
    return mb * W
end

## Number density
@inline function _number_density_accumulation(ra::NTuple{3, T}, rb::NTuple{3, T}, h :: T, smoothed_kernel :: Type{K}) where {T <: AbstractFloat}
    Ktyp = typeof(smoothed_kernel)
    W = Smoothed_kernel_function(Ktyp, ra, rb, h)
    return W
end

@inline function _number_density_accumulation(ra::NTuple{3, T}, rb::NTuple{3, T}, ha :: T, hb :: T, smoothed_kernel :: Type{K}) where {T <: AbstractFloat}
    Ktyp = typeof(smoothed_kernel)
    W = T(0.5) * (Smoothed_kernel_function(Ktyp, ra, rb, ha) + Smoothed_kernel_function(Ktyp, ra, rb, hb))
    return W
end

## Single quantity intepolation
@inline function _quantity_interpolate_accumulation(ra::NTuple{3, T}, rb::NTuple{3, T}, mb :: T, ρb :: T, Ab :: T, h :: T, smoothed_kernel :: Type{K}) where {T <: AbstractFloat}
    Ktyp = typeof(smoothed_kernel)
    W = Smoothed_kernel_function(Ktyp, ra, rb, h)
    mbWlρb = mb * W/ρb
    return Ab * mbWlρb
end

@inline function _quantity_interpolate_accumulation(ra::NTuple{3, T}, rb::NTuple{3, T}, mb :: T, ρb :: T, Ab :: T, ha :: T, hb :: T, smoothed_kernel :: Type{K}) where {T <: AbstractFloat}
    Ktyp = typeof(smoothed_kernel)
    W = T(0.5) * (Smoothed_kernel_function(Ktyp, ra, rb, ha) + Smoothed_kernel_function(Ktyp, ra, rb, hb))
    mbWlρb = mb * W/ρb
    return Ab * mbWlρb
end

@inline function _ShepardNormalization_accumulation(ra::NTuple{3, T}, rb::NTuple{3, T}, mb :: T, ρb :: T, h :: T, smoothed_kernel :: Type{K}) where {T <: AbstractFloat}
    Ktyp = typeof(smoothed_kernel)
    W = Smoothed_kernel_function(Ktyp, ra, rb, h)
    mbWlρb = mb * W/ρb
    return mbWlρb
end

@inline function _ShepardNormalization_accumulation(ra::NTuple{3, T}, rb::NTuple{3, T}, mb :: T, ρb :: T, ha :: T, hb :: T, smoothed_kernel :: Type{K}) where {T <: AbstractFloat}
    Ktyp = typeof(smoothed_kernel)
    W = T(0.5) * (Smoothed_kernel_function(Ktyp, ra, rb, ha) + Smoothed_kernel_function(Ktyp, ra, rb, hb))
    mbWlρb = mb * W/ρb
    return mbWlρb
end
