

function PhantomRevealer.to_MtlVector(input :: InterpolationInput)
    return InterpolationInput{Float32, MtlVector{Float32}, K, NCOLUMN}(
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
function Adapt.adapt_structure(to, x::PhantomRevealer.InterpolationInput)
    PhantomRevealer.InterpolationInput(
        x.Npart,
        Adapt.adapt(to, x.smoothed_kernel),
        Adapt.adapt(to, x.x),
        Adapt.adapt(to, x.y),
        Adapt.adapt(to, x.z),
        Adapt.adapt(to, x.m),
        Adapt.adapt(to, x.h),
        Adapt.adapt(to, x.ρ),
        ntuple(i->Adapt.adapt(to, x.quant[i]), length(x.quant))
    )
end