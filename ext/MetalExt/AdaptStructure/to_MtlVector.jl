

function PhantomRevealer.to_MtlVector(input :: InterpolationInput{T, V, K, NCOLUMN}) where {T <: AbstractFloat, V <: AbstractVector{T}, K <: AbstractSPHKernel, NCOLUMN}
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

function PhantomRevealer.to_MtlVector(enc :: MortonEncoding{D, TF, TI, VF, VI}) where {D, TF <: AbstractFloat, TI <: Unsigned, VF <: AbstractVector{TF}, VI <: AbstractVector{TI}}
    return MortonEncoding{D, Float32, TI, MtlVector{Float32}, MtlVector{TI}}(
        MtlVector{TI}(enc.order),
        MtlVector{TI}(enc.codes),
        ntuple(i -> MtlVector{Float32}(enc.coord[i]), D),
        MtlVector{Float32}(enc.h)
    )
end

function PhantomRevealer.to_MtlVector(brt :: BinaryRadixTree{V}) where {V <: AbstractVector{Int32}}
    return BinaryRadixTree{MtlVector{Int32}}(
        brt.root,
        brt.nleaf,
        MtlVector{Int32}(brt.left),
        MtlVector{Int32}(brt.right),
        MtlVector{Int32}(brt.escape),
        MtlVector{Int32}(brt.parent)
    )
end

function PhantomRevealer.to_MtlVector(AB :: AABB{D, TF, VF}) where {D, TF <: AbstractFloat, VF <: AbstractVector{TF}}
    return AABB{D, Float32, MtlVector{Float32}}(
        ntuple(i -> MtlVector{Float32}(AB.min[i]),D),
        ntuple(i -> MtlVector{Float32}(AB.max[i]),D)
    )

end

function PhantomRevealer.to_MtlVector(LBVH :: LinearBVH{D, TF, VF, VB}) where {D, TF <: AbstractFloat, VF <: AbstractVector{TF}, VB <: AbstractVector{Int32}}
    return LinearBVH{D, Float32, MtlVector{Float32}, MtlVector{Int32}}(
        to_MtlVector(LBVH.brt),
        ntuple(i -> MtlVector{Float32}(LBVH.leaf_coor[i]), D),
        MtlVector{Float32}(LBVH.leaf_h),
        to_MtlVector(LBVH.node_aabb),
        MtlVector{Float32}(LBVH.node_hmax)
    )
end

function PhantomRevealer.to_MtlVector(grid :: PointSamples{D, TF, VG, VC}) where {D, TF <: AbstractFloat, VG <: AbstractVector{TF}, VC <: NTuple{D, Vector{TF}}}
    return PointSamples{D, Float32, MtlVector{Float32}, NTuple{D, MtlVector{Float32}}}(
        MtlVector{Float32}(grid.grid),
        ntuple(i -> MtlVector{Float32}(grid.coor[i]), D)
    )
end

function PhantomRevealer.to_MtlVector(grid :: LineSamples{D, TF, VG, VC}) where {D, TF <: AbstractFloat, VG <: AbstractVector{TF}, VC <: NTuple{D, Vector{TF}}}
    return LineSamples{D, Float32, MtlVector{Float32}, NTuple{D, MtlVector{Float32}}}(
        MtlVector{Float32}(grid.grid),
        ntuple(i -> MtlVector{Float32}(grid.origin[i]), D),
        ntuple(i -> MtlVector{Float32}(grid.direction[i]), D)
    )
end

function PhantomRevealer.to_MtlVector(grid :: StructuredGrid{D, TF, V, A}) where {D, TF <: AbstractFloat, V <: AbstractVector{TF}, A <: AbstractArray{TF, D}}
    return StructuredGrid{D, Float32, MtlVector{Float32}, MtlArray{Float32, D}}(
        MtlArray{Float32, D}(grid.grid),
        ntuple(i -> MtlVector{Float32}(grid.axes[i]), D),
        grid.size
    )
end
