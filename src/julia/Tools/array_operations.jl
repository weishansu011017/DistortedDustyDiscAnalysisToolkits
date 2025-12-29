"""
Useful operations for array
    by Wei-Shan Su,
    September 23, 2025
"""

"""
    meshgrid(arrays :: AbstractVector...)

Create N meshgrid arrays from N 1D coordinate vectors.

# Parameters
- `arrays :: AbstractVector...` : One or more 1D coordinate vectors.

# Returns
- `NTuple{N, Array{T, N}}` : An N-tuple of N-dimensional arrays.  
  The `i`-th array varies along dimension `i` and is constant along other dimensions.
  Element type `T` follows the corresponding input vector’s element type.

# Notes
- Allocates O(∏ length(arrays[i])) elements per returned array.
- Output uses column-major ordering (Julia default).

# Examples
```julia
x = LinRange(0, 10, 250)
y = LinRange(0, 2π, 301)
X, Y = meshgrid(x, y)
```
"""
function meshgrid(arrays :: AbstractVector...)
    nd = length(arrays)
    grids = ntuple(i->repeat(reshape(arrays[i], ntuple(d->d==i ? length(arrays[i]) : 1,nd)...),ntuple(d->d==i ? 1 : length(arrays[d]), nd)...), nd)
    return grids 
end

"""
    reduce_mean(array :: AbstractArray, dim :: Int=1)

Compute the mean of `array` along the specified dimension, and drop that
dimension from the result.

# Parameters
- `array::AbstractArray` : Input array.
- `dim::Int=1` : Dimension along which the mean is computed.

# Returns
- `AbstractArray` : Array with one less dimension, after averaging.
"""
function reduce_mean(array :: AbstractArray, dim :: Int=1)
    return dropdims(mean(array; dims=dim), dims=dim)
end

"""
    marco _def_nanfunc(funcname)

Define a new `nan*` variant of the statistical function that ignores `NaN` values.

This macro generates two method overloads:
- `nanfunc(A::AbstractArray{T})`: returns the scalar result after removing all `NaN`s.
- `nanfunc(A::AbstractArray{T}, dims::Integer)`: performs reduction along the specified dimension, skipping `NaN`s and returning `T(NaN)` if the slice is entirely `NaN`.

# Parameters
- `funcname::Symbol`: The base function name (e.g., `:mean`, `:maximum`) to wrap.

# Generated
- A new function named `nanfuncname` will be defined in the current scope.
"""
macro _def_nanfunc(funcname)
    fname = Symbol(:nan, funcname)          
    quote
        @inline function $(fname)(A::AbstractArray{T}) where {T<:Number}
            vals = filter(!isnan, A)
            return isempty(vals) ? T(NaN) : $(funcname)(vals)
        end

        @inline function $(fname)(A::AbstractArray{T}, dims::Integer) where {T<:Number}
            if ndims(A) == 1
                return [nanmean(A)]
            else
                result = map(x -> isempty(x) ? T(NaN) : mean(x), eachslice(A, dims))
                return dropdims(result, dims)
            end
        end
    end |> esc        
end


@_def_nanfunc mean
@_def_nanfunc median
@_def_nanfunc std
@_def_nanfunc maximum
@_def_nanfunc minimum


"""
    Euclidean_distance(x :: V, y :: V, z :: V, ref :: NTuple{3, TF}) where {TF <: AbstractFloat, V <: AbstractVector}

Compute the Euclidean distance in 3D space between vectors `(x, y, z)` and a fixed reference point.

# Parameters
- `x :: V` : Vector of x-coordinates.
- `y :: V` : Vector of y-coordinates.
- `z :: V` : Vector of z-coordinates.
- `ref :: NTuple{3, TF}` : Reference point `(x_ref, y_ref, z_ref)`.

# Returns
- `Vector{TF}` : Distances from each point `(x[i], y[i], z[i])` to the reference point.
"""
@inline function Euclidean_distance(x :: V, y :: V, z :: V, ref :: NTuple{3, TF}) where {TF <: AbstractFloat, V <: AbstractVector}
    xt, yt, zt = ref
    r = similar(x, TF)
    @inbounds @simd for i in eachindex(x, y, z, r)
        xi = TF(x[i])
        yi = TF(y[i])
        zi = TF(z[i])
        dx = xi - xt
        dy = yi - yt
        dz = zi - zt
        r[i] = sqrt(dx * dx + dy * dy + dz * dz)
    end
    return r
end

"""
    Euclidean_distance(x :: V, y :: V, ref :: NTuple{2, TF}) where {TF <: AbstractFloat, V <: AbstractVector}

Compute the Euclidean distance in 2D space between vectors `(x, y)` and a fixed reference point.

# Parameters
- `x :: V` : Vector of x-coordinates.
- `y :: V` : Vector of y-coordinates.
- `ref :: NTuple{2, TF}` : Reference point `(x_ref, y_ref)`.

# Returns
- `Vector{TF}` : Distances from each point `(x[i], y[i])` to the reference point.
"""
@inline function Euclidean_distance(x :: V, y :: V, ref :: NTuple{2, TF}) where {TF <: AbstractFloat, V <: AbstractVector}
    xt, yt = ref
    s = similar(x, TF)
    @inbounds @simd for i in eachindex(x, y, s)
        xi = TF(x[i])
        yi = TF(y[i])
        dx = xi - xt
        dy = yi - yt
        s[i] = sqrt(dx * dx + dy * dy)
    end
    return s
end

"""
    invert_order(order)

Compute the inverse permutation `invorder` such that `invorder[order[i]] = i`.

Given a permutation vector `order`, this function constructs its inverse, which
can be used to revert any array that has been reordered using `order`. If an
array `A` was permuted into `B` via `B = A[order]`, then the original array can
be recovered by `A = B[invorder]`.

# Parameters
- `order::AbstractVector{<:Integer}`: A permutation vector of length `n`. All
  indices must form a valid permutation of `1:n`.

# Returns
- `invorder::AbstractVector{<:Integer}`: The inverse permutation, satisfying
  `invorder[order[i]] = i` for all `i`.

"""
function invert_order(order :: V) where {TI <: Integer, V <: AbstractVector{TI}}
    n = length(order)
    invorder = similar(order)
    @inbounds for i in 1:n
        invorder[order[i]] = i
    end
    return invorder
end

