

function PhantomRevealer.to_HostVector(input :: InterpolationInput{T, V, K, NCOLUMN}) where {T <: Float32, V <: MtlVector{T}, K <: AbstractSPHKernel, NCOLUMN}
    return InterpolationInput{Float32, Vector{Float32}, K, NCOLUMN}(
        input.Npart, 
        input.smoothed_kernel,
        Vector{Float32}(input.x),
        Vector{Float32}(input.y),
        Vector{Float32}(input.z),
        Vector{Float32}(input.m),
        Vector{Float32}(input.h), 
        Vector{Float32}(input.ρ), 
        ntuple(i -> Vector{Float32}(input.quant[i]),NCOLUMN))
end

function PhantomRevealer.to_HostVector(enc :: MortonEncoding{D, TF, TI, VF, VI}) where {D, TF <: Float32, TI <: Unsigned, VF <: MtlVector{TF}, VI <: MtlVector{TI}}
    return MortonEncoding{D, Float32, TI, Vector{Float32}, Vector{TI}}(
        Vector{TI}(enc.order),
        Vector{TI}(enc.codes),
        ntuple(i -> Vector{Float32}(enc.coord[i]), D),
        Vector{Float32}(enc.h)
    )
end

function PhantomRevealer.to_HostVector(brt :: BinaryRadixTree{V}) where {V <: MtlVector{Int32}}
    return BinaryRadixTree{Vector{Int32}}(
        brt.root,
        brt.nleaf,
        Vector{Int32}(brt.left),
        Vector{Int32}(brt.right),
        Vector{Int32}(brt.escape),
        Vector{Int32}(brt.parent)
    )
end

function PhantomRevealer.to_HostVector(AB :: AABB{D, TF, VF}) where {D, TF <: Float32, VF <: MtlVector{TF}}
    return AABB{D, Float32, Vector{Float32}}(
        ntuple(i -> Vector{Float32}(AB.min[i]),D),
        ntuple(i -> Vector{Float32}(AB.max[i]),D)
    )

end

function PhantomRevealer.to_HostVector(LBVH :: LinearBVH{D, TF, VF, VB}) where {D, TF <: Float32, VF <: MtlVector{TF}, VB <: MtlVector{Int32}}
    return LinearBVH{D, Float32, Vector{Float32}, Vector{Int32}}(
        to_HostVector(LBVH.brt),
        ntuple(i -> Vector{Float32}(LBVH.leaf_coor[i]), D),
        Vector{Float32}(LBVH.leaf_h),
        to_HostVector(LBVH.node_aabb),
        Vector{Float32}(LBVH.node_hmax)
    )
end

function PhantomRevealer.to_HostVector(grid :: PointSamples{D, TF, VG, VC}) where {D, TF <: Float32, VG <: MtlVector{TF}, VC <: NTuple{D, MtlVector{TF}}}
    return PointSamples(
        Vector{Float32}(grid.grid),
        ntuple(i -> Vector{Float32}(grid.coor[i]), D)
    )
end

function PhantomRevealer.to_HostVector(grid :: LineSamples{D, TF, VG, VC}) where {D, TF <: Float32, VG <: MtlVector{TF}, VC <: NTuple{D, MtlVector{TF}}}
    return LineSamples(
        Vector{Float32}(grid.grid),
        ntuple(i -> Vector{Float32}(grid.origin[i]), D),
        ntuple(i -> Vector{Float32}(grid.direction[i]), D)
    )
end

function PhantomRevealer.to_HostVector(grid :: StructuredGrid{D, TF, V, A}) where {D, TF <: Float32, V <: MtlVector{TF}, A <: MtlArray{TF, D}}
    return StructuredGrid{D, Float32, Vector{Float32}, Array{Float32, D}}(
        Array{Float32, D}(grid.grid),
        ntuple(i -> Vector{Float32}(grid.axes[i]), D),
        grid.size
    )
end
