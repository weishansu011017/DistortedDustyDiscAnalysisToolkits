
function PhantomRevealer.LineSamples(xo :: V, yo :: V, zo :: V, xd :: V, yd :: V, zd :: V, ::MetalComputeBackend) where {T <: AbstractFloat, V <: AbstractVector{T}}
    origin = (MtlVector{Float32}(xo), MtlVector{Float32}(yo), MtlVector{Float32}(zo))
    direction = (MtlVector{Float32}(xd), MtlVector{Float32}(yd), MtlVector{Float32}(zd))
    N = length(xo)
    vals = Metal.zeros(Float32, N)
    return LineSamples(vals, origin, direction)
end

function PhantomRevealer.LineSamples(xo :: V, yo :: V, xd :: V, yd :: V, ::MetalComputeBackend) where {T <: AbstractFloat, V <: AbstractVector{T}}
    origin = (MtlVector{Float32}(xo), MtlVector{Float32}(yo))
    direction = (MtlVector{Float32}(xd), MtlVector{Float32}(yd))
    N = length(xo)
    vals = Metal.zeros(Float32, N)
    return LineSamples(vals, origin, direction)
end
