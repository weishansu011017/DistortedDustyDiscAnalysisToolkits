

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
    )
end

function PhantomRevealer.to_MtlVector(brt :: BinaryRadixTree{TI, VI, A, B}) where {TI <: Unsigned, VI <: AbstractVector{TI},  A <: AbstractVector{Int}, B <: AbstractVector{Bool}}
    return BinaryRadixTree{TI, MtlVector{TI}, MtlVector{Int}, MtlVector{Bool}}(
        MtlVector{Int}(brt.left_child),
        MtlVector{Int}(brt.right_child),
        MtlVector{Bool}(brt.is_leaf_left),
        MtlVector{Bool}(brt.is_leaf_right),
        MtlVector{TI}(brt.leaf_parent),
        MtlVector{TI}(brt.node_parent),
        MtlVector{TI}(brt.visit_counter)
    )
end

function PhantomRevealer.to_MtlVector(AB :: AABB{D, TF, VF}) where {D, TF <: AbstractFloat, VF <: AbstractVector{TF}}
    return AABB{D, Float32, MtlVector{Float32}}(
        ntuple(i -> MtlVector{Float32}(AB.min[i]),D),
        ntuple(i -> MtlVector{Float32}(AB.max[i]),D)
    )

end

function PhantomRevealer.to_MtlVector(LBVH :: LinearBVH{D, TF, TI, VF, VI, A, B}) where {D, TF <: AbstractFloat, TI <: Unsigned, VF <: AbstractVector{TF}, VI <: AbstractVector{TI}, A <: AbstractVector{Int}, B <: AbstractVector{Bool}}
    return LinearBVH{D, Float32, TI, MtlVector{Float32}, MtlVector{TI}, MtlVector{Int}, MtlVector{Bool}}(
        to_MtlVector(LBVH.enc),
        to_MtlVector(LBVH.brt),
        to_MtlVector(LBVH.leaf_aabb),
        to_MtlVector(LBVH.node_aabb),
        MtlVector{Float32}(LBVH.node_hmax),
        LBVH.root
    )
end

function PhantomRevealer.to_MtlVector(grid :: GeneralGrid{D, TF, VG, VC}) where {D, TF <: AbstractFloat, VG <: AbstractVector{TF}, VC <: AbstractVector{NTuple{D, TF}}}
    return GeneralGrid{D, Float32, MtlVector{Float32}, MtlVector{NTuple{D, Float32}}}(
        MtlVector{Float32}(grid.grid),
        MtlVector{NTuple{D, Float32}}(grid.coor)
    )
end

function PhantomRevealer.to_MtlVector(grid :: StructuredGrid{D, TF, V, A}) where {D, TF <: AbstractFloat, V <: AbstractVector{TF}, A <: AbstractArray{TF, D}}
    return StructuredGrid{D, Float32, MtlVector{Float32}, MtlArray{Float32, D}}(
        MtlArray{Float32, D}(grid.grid),
        ntuple(i -> MtlVector{Float32}(grid.axes[i]), D),
        grid.size
    )
end