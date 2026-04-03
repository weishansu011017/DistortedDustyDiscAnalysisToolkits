
function Partia.PointSamples(x :: V, y :: V, z :: V, ::MetalComputeBackend) where {T <: AbstractFloat, V <: AbstractVector{T}}
    coords = (MtlVector{Float32}(x), MtlVector{Float32}(y), MtlVector{Float32}(z))
    N = length(x)
    vals = Metal.zeros(Float32, N)
    return PointSamples(vals, coords)
end
