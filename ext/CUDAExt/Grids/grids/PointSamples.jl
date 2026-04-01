
function PhantomRevealer.PointSamples(x :: V, y :: V, z :: V, ::CUDAComputeBackend) where {T <: AbstractFloat, V <: AbstractVector{T}}
    coords = (CuVector(x), CuVector(y), CuVector(z))
    N = length(x)
    vals = CUDA.zeros(T, N)
    return PointSamples{3, T, CuVector{T}, NTuple{3, CuVector{T}}}(vals, coords)
end
