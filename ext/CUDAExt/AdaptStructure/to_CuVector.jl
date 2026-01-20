

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

function PhantomRevealer.to_CuVector(enc :: MortonEncoding{D, TF, TI, VF, VI}) where {D, TF <: AbstractFloat, TI <: Unsigned, VF <: AbstractVector{TF}, VI <: AbstractVector{TI}}
    return MortonEncoding{D, TF, TI, CuVector{TF}, CuVector{TI}}(
        CuVector{TI}(enc.order),
        CuVector{TI}(enc.codes),
        ntuple(i -> CuVector{TF}(enc.coord[i]), D),
        CuVector{Float32}(enc.h)
    )
end

function PhantomRevealer.to_CuVector(brt :: BinaryRadixTree{V}) where {V <: AbstractVector{Int32}}
    return BinaryRadixTree{CuVector{Int32}}(
        brt.root,
        brt.nleaf,
        CuVector{Int32}(brt.left),
        CuVector{Int32}(brt.right),
        CuVector{Int32}(brt.escape),
        CuVector{Int32}(brt.parent)
    )
end

function PhantomRevealer.to_CuVector(AB :: AABB{D, TF, VF}) where {D, TF <: AbstractFloat, VF <: AbstractVector{TF}}
    return AABB{D, TF, CuVector{TF}}(
        ntuple(i -> CuVector{TF}(AB.min[i]),D),
        ntuple(i -> CuVector{TF}(AB.max[i]),D)
    )

end

function PhantomRevealer.to_CuVector(LBVH :: LinearBVH{D, TF, VF, VB}) where {D, TF <: AbstractFloat, VF <: AbstractVector{TF}, VB <: AbstractVector{Int32}}
    return LinearBVH{D, TF, CuVector{TF}, CuVector{Int32}}(
        to_CuVector(LBVH.brt),
        to_CuVector(LBVH.leaf_aabb),
        CuVector{TF}(LBVH.leaf_h),
        to_CuVector(LBVH.node_aabb),
        CuVector{TF}(LBVH.node_hmax)
    )
end

# Internal sanity check: host → CUDA → host round-trip for BRT/LBVH
function _cuda_roundtrip_structs_ok()
    (isdefined(CUDA, :has_cuda) && CUDA.has_cuda()) || return false

    # Minimal BinaryRadixTree (nleaf = 2 ⇒ total = 3)
    left = Int32[0, 2, 0]; right = Int32[0, 3, 0]; escape = Int32[0, 0, 0]; parent = Int32[0, 0, 1]
    brt_h = BinaryRadixTree{Vector{Int32}}(Int32(1), 2, left, right, escape, parent)
    brt_d = to_CuVector(brt_h)
    brt_rt = to_HostVector(brt_d)

    @assert brt_rt.root == brt_h.root
    @assert brt_rt.nleaf == brt_h.nleaf
    @assert brt_rt.left == brt_h.left && brt_rt.right == brt_h.right
    @assert brt_rt.escape == brt_h.escape && brt_rt.parent == brt_h.parent

    # Minimal LinearBVH (D = 3, nleaf = 2, ninternal = 1)
    leaf_min = ntuple(_ -> Float32[0, 1], 3); leaf_max = ntuple(_ -> Float32[0, 1], 3)
    node_min = ntuple(_ -> Float32[0], 3); node_max = ntuple(_ -> Float32[0], 3)
    leaf_h = Float32[0.1f0, 0.2f0]; node_hmax = Float32[0.2f0]
    leaf_aabb = AABB(leaf_min, leaf_max)
    node_aabb = AABB(node_min, node_max)
    lbvh_h = LinearBVH{3, Float32, Vector{Float32}, Vector{Int32}}(brt_h, leaf_aabb, leaf_h, node_aabb, node_hmax)
    lbvh_d = to_CuVector(lbvh_h)
    lbvh_rt = to_HostVector(lbvh_d)

    @assert lbvh_rt.brt.root == lbvh_h.brt.root
    @assert lbvh_rt.leaf_h == lbvh_h.leaf_h && lbvh_rt.node_hmax == lbvh_h.node_hmax
    @assert length(lbvh_rt.leaf_aabb.min[1]) == lbvh_h.brt.nleaf
    @assert length(lbvh_rt.node_aabb.min[1]) == lbvh_h.brt.nleaf - 1

    return true
end

function PhantomRevealer.to_CuVector(grid :: GeneralGrid{D, TF, VG, VC}) where {D, TF <: AbstractFloat, VG <: AbstractVector{TF}, VC <: NTuple{D, Vector{TF}}}
    return GeneralGrid{D, TF, CuVector{TF}, NTuple{D, CuVector{TF}}}(
        CuVector{TF}(grid.grid),
        ntuple(i -> CuVector{TF}(grid.coor[i]), D)
    )
end

function PhantomRevealer.to_CuVector(grid :: StructuredGrid{D, TF, V, A}) where {D, TF <: AbstractFloat, V <: AbstractVector{TF}, A <: AbstractArray{TF, D}}
    return StructuredGrid{D, TF, CuVector{TF}, CuArray{TF, D}}(
        CuArray{TF, D}(grid.grid),
        ntuple(i -> CuVector{TF}(grid.axes[i]), D),
        grid.size
    )
end