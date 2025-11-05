
struct MortonEncoding{D, TF <: AbstractFloat, TI <: Unsigned, V <: AbstractVector{TI}}
    order :: V          # Order of corresponding particles
    codes :: V          # Morton code
    ΔL       :: NTuple{D, TF}
    amin     :: NTuple{D, TF}
end

# Encoding & decoding morton code 
## The 3D morton code we used is (UInt64)(0(x0)(y0)(z0)(x1)(y1)......(z20)(x21)(y21)(z21)) or (UInt32)(00(x0)(y0)(z0)...(z9)(x10)(y10)(z10))
@inline function _expand_bits3D(x::UInt32)
    # Copied from https://stackoverflow.com/questions/1024754/how-to-compute-a-3d-morton-number-interleave-the-bits-of-3-ints
    x = (x | (x << 16)) & 0x30000ff   
    x = (x | (x << 8)) & 0x300f00f
    x = (x | (x << 4)) & 0x30c30c3
    x = (x | (x << 2)) & 0x9249249
    return x
end

@inline function _expand_bits3D(x::UInt64)
    # Copied from https://stackoverflow.com/questions/1024754/how-to-compute-a-3d-morton-number-interleave-the-bits-of-3-ints
    x = (x | (x << 32)) & 0x1f00000000ffff
    x = (x | (x << 16)) & 0x1f0000ff0000ff
    x = (x | (x << 8))  & 0x100f00f00f00f00f
    x = (x | (x << 4))  & 0x10c30c30c30c30c3
    x = (x | (x << 2))  & 0x1249249249249249
    return x
end

@inline function _compact_bits3D(x::UInt32)
    x &= 0x09249249
    x = (x ⊻ (x >>  2)) & 0x030C30C3
    x = (x ⊻ (x >>  4)) & 0x0300F00F
    x = (x ⊻ (x >>  8)) & 0xFF0000FF
    x = (x ⊻ (x >> 16)) & 0x000003FF
    return x
end

@inline function _compact_bits3D(x::UInt64)
    x &= 0x1249249249249249
    x = (x ⊻ (x >> 2)) & 0x10c30c30c30c30c3
    x = (x ⊻ (x >> 4)) & 0x100f00f00f00f00f
    x = (x ⊻ (x >> 8)) & 0x1f0000ff0000ff
    x = (x ⊻ (x >> 16)) & 0x1f00000000ffff
    x = (x ⊻ (x >> 32)) & 0x1fffff
    return x
end

## The 2D morton code we used is ((x0)(y0)(x1)(y1)....)
@inline function _expand_bits2D(x::UInt32)
    x = (x | (x << 8))  & 0x00FF00FF
    x = (x | (x << 4))  & 0x0F0F0F0F
    x = (x | (x << 2))  & 0x33333333
    x = (x | (x << 1))  & 0x55555555
    return x
end

@inline function _expand_bits2D(x::UInt64)
    x = (x | (x << 16)) & 0x0000FFFF0000FFFF
    x = (x | (x << 8))  & 0x00FF00FF00FF00FF
    x = (x | (x << 4))  & 0x0F0F0F0F0F0F0F0F
    x = (x | (x << 2))  & 0x3333333333333333
    x = (x | (x << 1))  & 0x5555555555555555
    return x
end

@inline function _compact_bits2D(x::UInt32)
    x &= 0x55555555
    x = (x ⊻ (x >> 1)) & 0x33333333
    x = (x ⊻ (x >> 2)) & 0x0F0F0F0F
    x = (x ⊻ (x >> 4)) & 0x00FF00FF
    x = (x ⊻ (x >> 8)) & 0x0000FFFF
    return x
end

@inline function _compact_bits2D(x::UInt64)
    x &= 0x5555555555555555
    x = (x ⊻ (x >> 1)) & 0x3333333333333333
    x = (x ⊻ (x >> 2)) & 0x0F0F0F0F0F0F0F0F
    x = (x ⊻ (x >> 4)) & 0x00FF00FF00FF00FF
    x = (x ⊻ (x >> 8)) & 0x0000FFFF0000FFFF
    x = (x ⊻ (x >> 16)) & 0x00000000FFFFFFFF
    return x
end

@inline function _encode_morton_code3D(ix :: T, iy :: T, iz :: T) where {T <: Unsigned}
    ex = _expand_bits3D(ix)
    ey = _expand_bits3D(iy)
    ez = _expand_bits3D(iz)
    return (ex << 2) | (ey << 1) | (ez << 0)
end

@inline function _decode_morton_code3D(code :: T) where {T <: Unsigned}
    x = _compact_bits3D(code >> 2)
    y = _compact_bits3D(code >> 1)
    z = _compact_bits3D(code >> 0)
    return (x, y, z)
end

function _encode_morton_code3D(ix :: V, iy :: V, iz :: V) where {T <: Unsigned, V <: AbstractVector{T}}
    code  = similar(ix)
    order = similar(ix)
    @inbounds @simd for i in eachindex(ix, iy, iz)
        ixi = ix[i]; iyi = iy[i]; izi = iz[i]
        codei = _encode_morton_code3D(ixi, iyi, izi)

        code[i] = codei
        order[i] = i
    end
    return code, order
end

function _decode_morton_code3D(code :: V) where {T <: Unsigned, V <: AbstractVector{T}}
    ix = similar(code)
    iy = similar(code)
    iz = similar(code)
    @inbounds @simd for i in eachindex(ix, iy, iz)
        codei = code[i]
        ixi, iyi, izi = _decode_morton_code3D(codei)

        @inbounds begin
            ix[i] = ixi
            iy[i] = iyi
            iz[i] = izi
        end
    end
    return ix, iy, iz
end

@inline function _encode_morton_code2D(ix::T, iy::T) where {T<:Unsigned}
    return (_expand_bits2D(ix) << 1) | _expand_bits2D(iy)
end

@inline function _decode_morton_code2D(code::T) where {T<:Unsigned}
    return (_compact_bits2D(code >> 1), _compact_bits2D(code))
end

function _encode_morton_code2D(ix :: V, iy :: V) where {T <: Unsigned, V <: AbstractVector{T}}
    code  = similar(ix)
    order = similar(ix)
    @inbounds @simd for i in eachindex(ix, iy)
        ixi = ix[i]; iyi = iy[i]
        codei = _encode_morton_code2D(ixi, iyi)

        code[i] = codei
        order[i] = i
    end
    return code, order
end

function _decode_morton_code2D(code :: V) where {T <: Unsigned, V <: AbstractVector{T}}
    ix = similar(code)
    iy = similar(code)
    @inbounds @simd for i in eachindex(ix, iy)
        codei = code[i]
        ixi, iyi = _decode_morton_code2D(codei)

        @inbounds begin
            ix[i] = ixi
            iy[i] = iyi
        end
    end
    return ix, iy
end

# AABB and distance for two points
@inline function _longest_common_prefix_length(a :: T, b :: T) where {T <: Unsigned}
    return leading_zeros(a ⊻ b)
end

@inline function _longest_common_prefix(a :: T, b :: T) where {T <: Unsigned}
    LCPL = _longest_common_prefix_length(a, b)
    β = sizeof(T) * 8 
    u = typemax(T)
    c = u << (β - LCPL)
    LCP = a & c
    return LCP
end

@inline function _prefix_range(a::T, b::T) where {T<:Unsigned}
    β = sizeof(T) * 8
    LCPL = _longest_common_prefix_length(a, b)
    p0 = _longest_common_prefix(a, b)
    p1 = p0 + (one(T) << (β - LCPL)) - one(T)
    return (p0, p1)
end

@inline function _prefix_size3D(a::T, b::T) where {T <: Unsigned}
    p0, p1 = _prefix_range(a, b)
    (x0, y0, z0) = _decode_morton_code3D(p0)
    (x1, y1, z1) = _decode_morton_code3D(p1)
    return (x1 - x0 + 1, y1 - y0 + 1, z1 - z0 + 1)
end

@inline function _prefix_size2D(a::T, b::T) where {T <: Unsigned}
    p0, p1 = _prefix_range(a, b)
    (x0, y0) = _decode_morton_code2D(p0)
    (x1, y1) = _decode_morton_code2D(p1)
    return (x1 - x0 + 1, y1 - y0 + 1)
end

# Projecting (x, y, z) to (ix, iy, iz)
@inline function _axis_bits(::Val{D}, ::Type{T}) where {D, T<:Unsigned}
    β = sizeof(T) * 8
    return (β - (β % D)) ÷ D
end

@inline function _axis_scale(::Val{D}, ::Type{T}, ::Type{TF}) where {D, T<:Unsigned, TF<:AbstractFloat}
    b = _axis_bits(Val(D), T)
    return exp2(TF(b)) - one(TF)
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

function _quantize_coords(x::V, y::V, z::V; CodeType :: Type{TI} = UInt64) where {TI <: Unsigned, T<:AbstractFloat, V<:AbstractVector{T}}
    # Normalize coordinate to [0, 1)
    fx = _normalize_vector(x)
    fy = _normalize_vector(y)
    fz = _normalize_vector(z)

    # Scaling the coordinate to [0, 2^k - 1)
    scale = _axis_scale(Val(3), CodeType, T)
    ix = similar(x, CodeType)
    iy = similar(y, CodeType)
    iz = similar(z, CodeType)
    @inbounds @simd for i in eachindex(ix, iy, iz)
        fxi = fx[i]; fyi = fy[i]; fzi = fz[i]

        ixi = CodeType(floor(scale * fxi))
        iyi = CodeType(floor(scale * fyi))
        izi = CodeType(floor(scale * fzi))

        @inbounds begin
            ix[i] = ixi
            iy[i] = iyi
            iz[i] = izi
        end
    end
    return ix, iy, iz
end

function _quantize_coords(x::V, y::V; CodeType :: Type{TI} = UInt64) where {TI <: Unsigned, T<:AbstractFloat, V<:AbstractVector{T}}
    # Normalize coordinate to [0, 1)
    fx = _normalize_vector(x)
    fy = _normalize_vector(y)

    # Scaling the coordinate to [0, 2^k - 1)
    scale = _axis_scale(Val(2), CodeType, T)
    ix = similar(x, CodeType)
    iy = similar(y, CodeType)
    @inbounds @simd for i in eachindex(ix, iy)
        fxi = fx[i]; fyi = fy[i]

        ixi = CodeType(floor(scale * fxi))
        iyi = CodeType(floor(scale * fyi))

        @inbounds begin
            ix[i] = ixi
            iy[i] = iyi
        end
    end
    return ix, iy
end

"""
    encoding_particles(x::V, y::V, z::V; CodeType::Type{TI}=UInt64)

Encode a set of 3D particle coordinates into Morton codes.

# Parameters
- `x, y, z :: AbstractVector{T}`: Particle positions along each axis (floating-point).
- `CodeType :: Type{TI}`: Unsigned integer type used for Morton encoding (`UInt32` or `UInt64`).

# Returns
- `MortonEncoding{3, T, TI, typeof(order)}`: Struct containing Morton codes, particle order, and bounding-box info.
"""
function encoding_particles(x :: V, y :: V, z :: V; CodeType :: Type{TI} = UInt64) where {TI <: Unsigned, T<:AbstractFloat, V<:AbstractVector{T}}
    amin = (minimum(x), minimum(y), minimum(z))
    xmin, ymin, zmin = amin
    ΔL = ((maximum(x) - xmin), (maximum(y) - ymin), (maximum(z) - zmin))

    ix, iy, iz = _quantize_coords(x, y, z, CodeType=CodeType)
    codes, order  = _encode_morton_code3D(ix, iy, iz)

    return MortonEncoding{3, T, TI, typeof(order)}(order, codes, ΔL, amin)
end

"""
    encoding_particles(x::V, y::V; CodeType::Type{TI}=UInt64)

Encode a set of 2D particle coordinates into Morton codes.

# Parameters
- `x, y :: AbstractVector{T}`: Particle positions along each axis (floating-point).
- `CodeType :: Type{TI}`: Unsigned integer type used for Morton encoding (`UInt32` or `UInt64`).

# Returns
- `MortonEncoding{2, T, TI, typeof(order)}`: Struct containing Morton codes, particle order, and bounding-box info.
"""
function encoding_particles(x :: V, y :: V; CodeType :: Type{TI} = UInt64) where {TI <: Unsigned, T<:AbstractFloat, V<:AbstractVector{T}}
    amin = (minimum(x), minimum(y))
    xmin, ymin = amin
    ΔL = ((maximum(x) - xmin), (maximum(y) - ymin))

    ix, iy = _quantize_coords(x, y, CodeType=CodeType)
    codes, order  = _encode_morton_code2D(ix, iy)

    return MortonEncoding{2, T, TI, typeof(order)}(order, codes, ΔL, amin)
end

"""
    decoding_particles(Encoding::MortonEncoding{3, T, TI})

Decode Morton codes back into 3D physical coordinates.

# Parameters
- `Encoding :: MortonEncoding{3, T, TI}`: Morton-encoded particle data.

# Returns
- `(x, y, z) :: NTuple{3, Vector{T}}`: Reconstructed particle positions.
"""
function decoding_particles(Encoding :: MortonEncoding{3, T, TI}) where {T <: AbstractFloat, TI <: Unsigned}
    ΔLx, ΔLy, ΔLz = Encoding.ΔL
    xmin, ymin, zmin = Encoding.amin
    codes = Encoding.codes

    scale = inv(_axis_scale(Val(3),TI, T))
    ix, iy, iz = _decode_morton_code3D(codes)

    x = similar(ix, T)
    y = similar(iy, T)
    z = similar(iz, T)
    @inbounds @simd for i in eachindex(x, y, z)
        ixi = ix[i]; iyi = iy[i]; izi = iz[i]
        fxi = ixi * scale
        fyi = iyi * scale
        fzi = izi * scale

        xi = (fxi * ΔLx) + xmin
        yi = (fyi * ΔLy) + ymin
        zi = (fzi * ΔLz) + zmin

        @inbounds begin
            x[i] = xi
            y[i] = yi
            z[i] = zi
        end
    end
    return x, y, z
end

function _decoding_particles(Encoding :: MortonEncoding{2, T, TI}) where {T <: AbstractFloat, TI <: Unsigned}
    ΔLx, ΔLy = Encoding.ΔL
    xmin, ymin = Encoding.amin
    codes = Encoding.codes

    scale = inv(_axis_scale(Val(2),TI, T))
    ix, iy = _decode_morton_code2D(codes)

    x = similar(ix, T)
    y = similar(iy, T)
    @inbounds @simd for i in eachindex(x, y)
        ixi = ix[i]; iyi = iy[i]
        fxi = ixi * scale
        fyi = iyi * scale

        xi = (fxi * ΔLx) + xmin
        yi = (fyi * ΔLy) + ymin

        @inbounds begin
            x[i] = xi
            y[i] = yi
        end
    end
    return x, y
end

"""
    sort_by_morton!(enc::MortonEncoding)

Sort particles by Morton code in-place.

# Parameters
- `enc :: MortonEncoding`: The encoding struct to be sorted.

# Returns
- `p :: Vector{Int}`: The permutation indices used for sorting.
"""
@inline function sort_by_morton!(enc::MortonEncoding)
    p = sortperm(enc.codes; alg=QuickSort)
    enc.codes .= enc.codes[p]
    enc.order .= enc.order[p]
    return p
end

