

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
        to_MtlVector(LBVH.leaf_aabb),
        MtlVector{Float32}(LBVH.leaf_h),
        to_MtlVector(LBVH.node_aabb),
        MtlVector{Float32}(LBVH.node_hmax)
    )
end

# Internal sanity check: host → Metal → host round-trip for BRT/LBVH
function _metal_roundtrip_structs_ok()
    (isdefined(Metal, :functional) && Metal.functional()) || return false

    left = Int32[0, 2, 0]; right = Int32[0, 3, 0]; escape = Int32[0, 0, 0]; parent = Int32[0, 0, 1]
    brt_h = BinaryRadixTree{Vector{Int32}}(Int32(1), 2, left, right, escape, parent)
    brt_d = to_MtlVector(brt_h)
    brt_rt = to_HostVector(brt_d)

    @assert brt_rt.root == brt_h.root
    @assert brt_rt.nleaf == brt_h.nleaf
    @assert brt_rt.left == brt_h.left && brt_rt.right == brt_h.right
    @assert brt_rt.escape == brt_h.escape && brt_rt.parent == brt_h.parent

    leaf_min = ntuple(_ -> Float32[0, 1], 3); leaf_max = ntuple(_ -> Float32[0, 1], 3)
    node_min = ntuple(_ -> Float32[0], 3); node_max = ntuple(_ -> Float32[0], 3)
    leaf_h = Float32[0.1f0, 0.2f0]; node_hmax = Float32[0.2f0]
    leaf_aabb = AABB(leaf_min, leaf_max)
    node_aabb = AABB(node_min, node_max)
    lbvh_h = LinearBVH{3, Float32, Vector{Float32}, Vector{Int32}}(brt_h, leaf_aabb, leaf_h, node_aabb, node_hmax)
    lbvh_d = to_MtlVector(lbvh_h)
    lbvh_rt = to_HostVector(lbvh_d)

    @assert lbvh_rt.brt.root == lbvh_h.brt.root
    @assert lbvh_rt.leaf_h == lbvh_h.leaf_h && lbvh_rt.node_hmax == lbvh_h.node_hmax
    @assert length(lbvh_rt.leaf_aabb.min[1]) == lbvh_h.brt.nleaf
    @assert length(lbvh_rt.node_aabb.min[1]) == lbvh_h.brt.nleaf - 1

    return true
end

function PhantomRevealer.to_MtlVector(grid :: GeneralGrid{D, TF, VG, VC}) where {D, TF <: AbstractFloat, VG <: AbstractVector{TF}, VC <: NTuple{D, Vector{TF}}}
    return GeneralGrid{D, Float32, MtlVector{Float32}, NTuple{D, MtlVector{Float32}}}(
        MtlVector{Float32}(grid.grid),
        ntuple(i -> MtlVector{Float32}(grid.coor[i]), D)
    )
end

function PhantomRevealer.to_MtlVector(grid :: StructuredGrid{D, TF, V, A}) where {D, TF <: AbstractFloat, V <: AbstractVector{TF}, A <: AbstractArray{TF, D}}
    return StructuredGrid{D, Float32, MtlVector{Float32}, MtlArray{Float32, D}}(
        MtlArray{Float32, D}(grid.grid),
        ntuple(i -> MtlVector{Float32}(grid.axes[i]), D),
        grid.size
    )
end