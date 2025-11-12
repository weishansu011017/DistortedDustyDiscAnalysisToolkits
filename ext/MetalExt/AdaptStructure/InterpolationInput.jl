

function PhantomRevealer.to_MtlVector(input :: InterpolationInput{T, V, K, NCOLUMN}) where {T <: AbstractFloat, V <: AbstractVector{T}, K <: AbstractSPHKernel, NCOLUMN}
    return InterpolationInput{Float32, MtlVector{T}, K, NCOLUMN}(
        input.Npart, 
        input.smoothed_kernel,
        MtlVector{Float32}(input.x),
        MtlVector{Float32}(input.y),
        MtlVector{Float32}(input.z),
        MtlVector{Float32}(input.m),
        MtlVector{Float32}(input.h), 
        MtlVector{Float32}(input.ρ), 
        ntuple(i -> MtlVector{Float32}(input.quant[i]),NCOLUMN))
end