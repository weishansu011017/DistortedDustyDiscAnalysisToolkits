
#### I don't know... Maybe next time. TToooooo busy

@inline function _quantize_coords!(ix :: V, iy :: V, iz :: V, fvs :: NTuple{3, V}, x::V, y::V, z::V; CodeType :: Type{TI} = UInt64) where {TI <: Unsigned, T<:AbstractFloat, V<:CuVector{T}}
    # Scaling the coordinate to [0, 2^k - 1)
    scale = _axis_scale(Val(3), CodeType, T)
    
    # Normalize coordinate to [0, 1)
    fx = fvs[1]
    fy = fvs[2]
    fz = fvs[3]
    _normalize_vector!(fx, x)
    _normalize_vector!(fy, y)
    _normalize_vector!(fz, z)

    @inbounds @threads for i in eachindex(ix, iy, iz)
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
    return nothing
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
    Base.permute!(enc.codes, p)
    Base.permute!(enc.order, p)
    for dir in enc.coord
        Base.permute!(dir, p)
    end
    @inbounds for i in 2:length(enc.codes)
        if enc.codes[i] <= enc.codes[i - 1]
            enc.codes[i] = enc.codes[i-1] + one(eltype(enc.codes))
        end
    end
    return p
end


"""
    MortonEncoding(x::V, y::V, z::V; CodeType::Type{TI}=UInt64)

Encode a set of 3D particle coordinates into Morton codes.

# Parameters
- `x, y, z :: AbstractVector{T}`: Particle positions along each axis (floating-point).
- `CodeType :: Type{TI}`: Unsigned integer type used for Morton encoding (`UInt32` or `UInt64`).

# Returns
- `MortonEncoding{3, T, TI, V, typeof(order)}`: Struct containing Morton codes, particle order, and coordinates, ordered by Morton codes
"""
function MortonEncoding(x :: V, y :: V, z :: V; CodeType :: Type{TI} = UInt64) where {TI <: Unsigned, T<:AbstractFloat, V<:CuVector{T}}
    ix = CUDA.zeros(TI, length(x))
    iy = CUDA.zeros(TI, length(y))
    iz = CUDA.zeros(TI, length(z))
    _quantize_coords!(x, y, z, ix, iy, iz, CodeType=CodeType)
    codes, order  = _encode_morton_code3D(ix, iy, iz)
    enc = MortonEncoding{3, T, TI, V, typeof(order)}(order, codes, (x, y, z))
    sort_by_morton!(enc)
    return enc
end

"""
    MortonEncoding(x::V, y::V; CodeType::Type{TI}=UInt64)

Encode a set of 2D particle coordinates into Morton codes.

# Parameters
- `x, y :: AbstractVector{T}`: Particle positions along each axis (floating-point).
- `CodeType :: Type{TI}`: Unsigned integer type used for Morton encoding (`UInt32` or `UInt64`).

# Returns
- `MortonEncoding{2, T, TI, V, typeof(order)}`: Struct containing Morton codes, particle order, and coordinates, ordered by Morton codes
"""
function MortonEncoding(x :: V, y :: V; CodeType :: Type{TI} = UInt64) where {TI <: Unsigned, T<:AbstractFloat, V<:AbstractVector{T}}
    xcopy = copy(x); ycopy = copy(y)
    ix, iy = _quantize_coords(xcopy, ycopy, CodeType=CodeType)
    codes, order  = _encode_morton_code2D(ix, iy)
    enc = MortonEncoding{2, T, TI, V, typeof(order)}(order, codes, (xcopy, ycopy))
    sort_by_morton!(enc)
    return enc
end