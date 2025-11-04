
struct MortonEncoding{TI <: Unsigned, V <: AbstractVector{TI}}
    codes :: V
    bitdepth :: Int
end

@inline function _get_href(h :: V, qlevel :: S) where {S <: AbstractFloat, T <: AbstractFloat, V <: AbstractVector{T}}
    return quantile(h, qlevel)
end

@inline function _get_ΔL(x :: V, y :: V, z :: V) where {T<: AbstractFloat, V <: AbstractVector{T}}
    return max((maximum(x) - minimum(x)),
               (maximum(y) - minimum(y)),
               (maximum(z) - minimum(z))
               )
end

@inline function _estimate_bit_depth(ΔL :: T, href :: T) where {T <: AbstractFloat}
    return Int(ceil(log2(ΔL / href)))
end

@inline function _estimate_bit_depth(ΔL :: T, href :: S) where {T <: AbstractFloat, S <:AbstractFloat}
    promoted = promote(ΔL, href)
    return _estimate_bit_depth(promoted...)
end

function _normalize_vector(v :: V) where {T <: AbstractFloat, V <: AbstractVector{T}}
    vmin = minimum(v); vmax = maximum(v)
    Δv = vmax - vmin
    invΔv = inv(Δv)
    normalized_v = similar(v)
    @inbounds @simd for i in eachindex(normalized_v)
        vi = v[i]
        normalized_vi = (vi - vmin) * invΔv
        normalized_v[i] = normalized_vi
    end
    return normalized_v
end

function _quantize_coords(x::V, y::V, z::V, b::Int; CodeType :: Type{TI} = UInt64) where {TI <: Unsigned, T<:AbstractFloat, V<:AbstractVector{T}}
    # Normalize coordinate to [0, 1)
    fx = _normalize_vector(x)
    fy = _normalize_vector(y)
    fz = _normalize_vector(z)

    # Scaling the coordinate to [0, 2^k - 1)
    scale = (T(2.0)^b - one(T))
    ix = similar(x, CodeType)
    iy = similar(y, CodeType)
    iz = similar(z, CodeType)
    @inbounds @simd for i in eachindex(ix, iy, iz)
        fxi = fx[i]; fyi = fy[i]; fzi = fz[i]

        ixi = CodeType(floor(scale * fxi))
        iyi = CodeType(floor(scale * fyi))
        izi = CodeType(floor(scale * fzi))

        ix[i] = ixi
        iy[i] = iyi
        iz[i] = izi
    end
    return ix, iy, iz
end

@inline function _expand_bits(x::UInt32)
    # Copy from https://stackoverflow.com/questions/1024754/how-to-compute-a-3d-morton-number-interleave-the-bits-of-3-ints
    x = (x | (x << 16)) & 0x30000ff   
    x = (x | (x << 8)) & 0x300f00f
    x = (x | (x << 4)) & 0x30c30c3
    x = (x | (x << 2)) & 0x9249249
    return x
end

@inline function _expand_bits(x::UInt64)
    # Copy from https://stackoverflow.com/questions/1024754/how-to-compute-a-3d-morton-number-interleave-the-bits-of-3-ints
    x = (x | (x << 32)) & 0x1f00000000ffff
    x = (x | (x << 16)) & 0x1f0000ff0000ff
    x = (x | (x << 8))  & 0x100f00f00f00f00f
    x = (x | (x << 4))  & 0x10c30c30c30c30c3
    x = (x | (x << 2))  & 0x1249249249249249
    return x
end

@inline function _morton_code(ix :: T, iy :: T, iz :: T, b :: Int) where {T <: Unsigned}
end



