
function PhantomRevealer.GeneralGrid(x :: V, y :: V, z :: V, ::MetalComputeBackend) where {T <: AbstractFloat, V <: AbstractVector{T}}
    coords = map((a,b,c)->(a,b,c), MtlVector{Float32}(x), MtlVector{Float32}(y), MtlVector{Float32}(z))
    N = length(x)
    vals = Metal.zeros(Float32, N)
    return GeneralGrid{3, Float32, MtlVector{Float32}, MtlVector{NTuple{3, Float32}}}(vals, coords)
end