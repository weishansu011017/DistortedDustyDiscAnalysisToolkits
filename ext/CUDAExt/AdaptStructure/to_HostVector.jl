

function PhantomRevealer.to_HostVector(input :: InterpolationInput{T, V, K, NCOLUMN}) where {T <: AbstractFloat, V <: CuVector{T}, K <: AbstractSPHKernel, NCOLUMN}
    return InterpolationInput{T, Vector{T}, K, NCOLUMN}(
        input.Npart, 
        input.smoothed_kernel,
        Vector{T}(input.x),
        Vector{T}(input.y),
        Vector{T}(input.z),
        Vector{T}(input.m),
        Vector{T}(input.h), 
        Vector{T}(input.ρ), 
        ntuple(i -> Vector{T}(input.quant[i]),NCOLUMN))
end

function PhantomRevealer.to_HostVector(enc :: MortonEncoding{D, TF, TI, VF, VI}) where {D, TF <: AbstractFloat, TI <: Unsigned, VF <: CuVector{TF}, VI <: CuVector{TI}}
    return MortonEncoding{D, TF, TI, Vector{TF}, Vector{TI}}(
        Vector{TI}(enc.order),
        Vector{TI}(enc.codes),
        ntuple(i -> Vector{TF}(enc.coord[i]), D),
        Vector{TF}(enc.h)
    )
end

function PhantomRevealer.to_HostVector(brt :: BinaryRadixTree{V}) where {V <: CuVector{Int32}}
    return BinaryRadixTree{Vector{Int32}}(
        brt.root,
        brt.nleaf,
        Vector{Int32}(brt.left),
        Vector{Int32}(brt.right),
        Vector{Int32}(brt.escape),
        Vector{Int32}(brt.parent)
    )
end

function PhantomRevealer.to_HostVector(AB :: AABB{D, TF, VF}) where {D, TF <: AbstractFloat, VF <: CuVector{TF}}
    return AABB{D, TF, Vector{TF}}(
        ntuple(i -> Vector{TF}(AB.min[i]),D),
        ntuple(i -> Vector{TF}(AB.max[i]),D)
    )

end

function PhantomRevealer.to_HostVector(LBVH :: LinearBVH{D, TF, VF, VB}) where {D, TF <: AbstractFloat, VF <: CuVector{TF}, VB <: CuVector{Int32}}
    return LinearBVH{D, TF, Vector{TF}, Vector{Int32}}(
        to_HostVector(LBVH.brt),
        ntuple(i -> Vector{TF}(LBVH.leaf_coor[i]), D),
        Vector{TF}(LBVH.leaf_h),
        to_HostVector(LBVH.node_aabb),
        Vector{TF}(LBVH.node_hmax)
    )
end

function PhantomRevealer.to_HostVector(grid :: GeneralGrid{D, TF, VG, VC}) where {D, TF <: AbstractFloat, VG <: CuVector{TF}, VC <: NTuple{D, CuVector{TF}}}
    return GeneralGrid{D, TF, Vector{TF}, NTuple{D, Vector{TF}}}(
        Vector{TF}(grid.grid),
        ntuple(i -> Vector{TF}(grid.coor[i]), D)
    )
end

function PhantomRevealer.to_HostVector(grid :: StructuredGrid{D, TF, V, A}) where {D, TF <: AbstractFloat, V <: CuVector{TF}, A <: CuArray{TF, D}}
    return StructuredGrid{D, TF, Vector{TF}, Array{TF, D}}(
        Array{TF, D}(grid.grid),
        ntuple(i -> Vector{TF}(grid.axes[i]), D),
        grid.size
    )
end
