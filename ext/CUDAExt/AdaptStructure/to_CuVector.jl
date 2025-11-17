

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
    )
end

function PhantomRevealer.to_CuVector(brt :: BinaryRadixTree{TI, VI, A, B}) where {TI <: Unsigned, VI <: AbstractVector{TI},  A <: AbstractVector{Int}, B <: AbstractVector{Bool}}
    return BinaryRadixTree{TI, CuVector{TI}, CuVector{Int}, CuVector{Bool}}(
        CuVector{Int}(brt.left_child),
        CuVector{Int}(brt.right_child),
        CuVector{Bool}(brt.is_leaf_left),
        CuVector{Bool}(brt.is_leaf_right),
        CuVector{TI}(brt.leaf_parent),
        CuVector{TI}(brt.node_parent),
        CuVector{TI}(brt.visit_counter)
    )
end

function PhantomRevealer.to_CuVector(AB :: AABB{D, TF, VF}) where {D, TF <: AbstractFloat, VF <: AbstractVector{TF}}
    return AABB{D, TF, CuVector{TF}}(
        ntuple(i -> CuVector{TF}(AB.min[i]),D),
        ntuple(i -> CuVector{TF}(AB.max[i]),D)
    )

end

function PhantomRevealer.to_CuVector(LBVH :: LinearBVH{D, TF, TI, VF, VI, A, B}) where {D, TF <: AbstractFloat, TI <: Unsigned, VF <: AbstractVector{TF}, VI <: AbstractVector{TI}, A <: AbstractVector{Int}, B <: AbstractVector{Bool}}
    return LinearBVH{D, TF, TI, CuVector{TF}, CuVector{TI}, CuVector{Int}, CuVector{Bool}}(
        to_CuVector(LBVH.enc),
        to_CuVector(LBVH.brt),
        to_CuVector(LBVH.leaf_aabb),
        to_CuVector(LBVH.node_aabb),
        LBVH.root
    )
end

function PhantomRevealer.to_CuVector(grid :: GeneralGrid{D, TF, VG, VC}) where {D, TF <: AbstractFloat, VG <: AbstractVector{TF}, VC <: AbstractVector{NTuple{D, TF}}}
    return GeneralGrid{D, TF, CuVector{TF}, CuVector{NTuple{D, TF}}}(
        CuVector{TF}(grid.grid),
        CuVector{NTuple{D, TF}}(grid.coor)
    )
end

function PhantomRevealer.to_CuVector(grid :: StructuredGrid{D, TF, V, A}) where {D, TF <: AbstractFloat, V <: AbstractVector{TF}, A <: AbstractArray{TF, D}}
    return StructuredGrid{D, TF, CuVector{TF}, CuArray{TF, D}}(
        CuArray{TF, D}(grid.grid),
        ntuple(i -> CuVector{TF}(grid.axes[i]), D),
        grid.size
    )
end