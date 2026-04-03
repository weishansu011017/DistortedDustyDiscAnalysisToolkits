

function Partia.to_CuVector(input :: InterpolationInput{D, T, V, K, NCOLUMN}) where {D, T <: AbstractFloat, V <: AbstractVector{T}, K <: AbstractSPHKernel, NCOLUMN}
    return InterpolationInput{D, T, CuVector{T}, K, NCOLUMN}(
        input.Npart,
        input.smoothed_kernel,
        ntuple(i -> CuVector{T}(input.coord[i]), Val(D)),
        CuVector{T}(input.m),
        CuVector{T}(input.h),
        CuVector{T}(input.ρ),
        ntuple(i -> CuVector{T}(input.quant[i]), Val(NCOLUMN))
    )
end

function Partia.to_CuVector(enc :: MortonEncoding{D, TF, TI, VF, VI}) where {D, TF <: AbstractFloat, TI <: Unsigned, VF <: AbstractVector{TF}, VI <: AbstractVector{TI}}
    return MortonEncoding{D, TF, TI, CuVector{TF}, CuVector{TI}}(
        CuVector{TI}(enc.order),
        CuVector{TI}(enc.codes),
        ntuple(i -> CuVector{TF}(enc.coord[i]), D),
        CuVector{Float32}(enc.h)
    )
end

function Partia.to_CuVector(brt :: BinaryRadixTree{V}) where {V <: AbstractVector{Int32}}
    return BinaryRadixTree{CuVector{Int32}}(
        brt.root,
        brt.nleaf,
        CuVector{Int32}(brt.left),
        CuVector{Int32}(brt.right),
        CuVector{Int32}(brt.escape),
        CuVector{Int32}(brt.parent)
    )
end

function Partia.to_CuVector(AB :: AABB{D, TF, VF}) where {D, TF <: AbstractFloat, VF <: AbstractVector{TF}}
    return AABB{D, TF, CuVector{TF}}(
        ntuple(i -> CuVector{TF}(AB.min[i]), D),
        ntuple(i -> CuVector{TF}(AB.max[i]), D)
    )
end

function Partia.to_CuVector(LBVH :: LinearBVH{D, TF, VF, VB}) where {D, TF <: AbstractFloat, VF <: AbstractVector{TF}, VB <: AbstractVector{Int32}}
    return LinearBVH{D, TF, CuVector{TF}, CuVector{Int32}}(
        to_CuVector(LBVH.brt),
        ntuple(i -> CuVector{TF}(LBVH.leaf_coor[i]), D),
        CuVector{TF}(LBVH.leaf_h),
        to_CuVector(LBVH.node_aabb),
        CuVector{TF}(LBVH.node_hmax)
    )
end

function Partia.to_CuVector(grid :: PointSamples{D, TF, VG, VC}) where {D, TF <: AbstractFloat, VG <: AbstractVector{TF}, VC <: NTuple{D, Vector{TF}}}
    return PointSamples(
        CuVector{TF}(grid.grid),
        ntuple(i -> CuVector{TF}(grid.coor[i]), D)
    )
end

function Partia.to_CuVector(grid :: LineSamples{D, TF, VG, VC}) where {D, TF <: AbstractFloat, VG <: AbstractVector{TF}, VC <: NTuple{D, Vector{TF}}}
    return LineSamples(
        CuVector{TF}(grid.grid),
        ntuple(i -> CuVector{TF}(grid.origin[i]), D),
        ntuple(i -> CuVector{TF}(grid.direction[i]), D)
    )
end

function Partia.to_CuVector(grid :: StructuredGrid{D, TF, V, A}) where {D, TF <: AbstractFloat, V <: AbstractVector{TF}, A <: AbstractArray{TF, D}}
    return StructuredGrid{D, TF, CuVector{TF}, CuArray{TF, D}}(
        CuArray{TF, D}(grid.grid),
        ntuple(i -> CuVector{TF}(grid.axes[i]), D),
        grid.size
    )
end
