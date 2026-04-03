
function Partia.LineSamples(xo :: V, yo :: V, zo :: V, xd :: V, yd :: V, zd :: V, :: CUDAComputeBackend) where {T <: AbstractFloat, V <: AbstractVector{T}}
    origin = (CuVector(xo), CuVector(yo), CuVector(zo))
    direction = (CuVector(xd), CuVector(yd), CuVector(zd))
    N = length(xo)
    vals = CUDA.zeros(T, N)
    return LineSamples(vals, origin, direction)
end

function Partia.LineSamples(xo :: V, yo :: V, xd :: V, yd :: V, :: CUDAComputeBackend) where {T <: AbstractFloat, V <: AbstractVector{T}}
    origin = (CuVector(xo), CuVector(yo))
    direction = (CuVector(xd), CuVector(yd))
    N = length(xo)
    vals = CUDA.zeros(T, N)
    return LineSamples(vals, origin, direction)
end
