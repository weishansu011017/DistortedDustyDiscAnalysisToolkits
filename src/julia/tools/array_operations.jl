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
