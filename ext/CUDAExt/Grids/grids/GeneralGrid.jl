
function PhantomRevealer.GeneralGrid(x :: V, y :: V, z :: V, ::CUDAComputeBackend) where {T <: AbstractFloat, V <: AbstractVector{T}}
    coords = map((a,b,c)->(a,b,c), CuVector(x), CuVector(y), CuVector(z))
    N = length(x)
    vals = CUDA.zeros(T, N)
    return GeneralGrid{3, T, CuVector{T}, CuVector{NTuple{3, T}}}(vals, coords)
end