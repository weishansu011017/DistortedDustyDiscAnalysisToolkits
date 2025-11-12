

function PhantomRevealer.to_CuVector(input :: InterpolationInput{T, V, K, NCOLUMN}) where {T <: AbstractFloat, V <: AbstractVector{T}, K <: AbstractSPHKernel, NCOLUMN}
    return InterpolationInput{T, CuVector{T}, K, NCOLUMN}(
        input.Npart, 
        input.smoothed_kernel,
        CuVector{T}(input.x),
        CuVector{T}(input.y),
        CuVector{T}(input.z),
        CuVector{T}(input.m),
        CuVector{T}(input.h), 
        CuVector{T}(input.ρ), 
        ntuple(i -> CuVector{T}(input.quant[i]),NCOLUMN))
end