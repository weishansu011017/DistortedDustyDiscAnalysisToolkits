

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

function PhantomRevealer.to_HostVector(brt :: BinaryRadixTree{TI, VI, A, B}) where {TI <: Unsigned, VI <: CuVector{TI},  A <: CuVector{Int}, B <: CuVector{Bool}}
    return BinaryRadixTree{TI, Vector{TI}, Vector{Int}, Vector{Bool}}(
        Vector{Int}(brt.left_child),
        Vector{Int}(brt.right_child),
        Vector{Bool}(brt.is_leaf_left),
        Vector{Bool}(brt.is_leaf_right),
        Vector{TI}(brt.leaf_parent),
        Vector{TI}(brt.node_parent),
        Vector{TI}(brt.visit_counter)
    )
end

function PhantomRevealer.to_HostVector(AB :: AABB{D, TF, VF}) where {D, TF <: AbstractFloat, VF <: CuVector{TF}}
    return AABB{D, TF, Vector{TF}}(
        ntuple(i -> Vector{TF}(AB.min[i]),D),
        ntuple(i -> Vector{TF}(AB.max[i]),D)
    )

end

function PhantomRevealer.to_HostVector(LBVH :: LinearBVH{D, TF, TI, VF, VI, A, B}) where {D, TF <: AbstractFloat, TI <: Unsigned, VF <: CuVector{TF}, VI <: CuVector{TI}, A <: CuVector{Int}, B <: CuVector{Bool}}
    return LinearBVH{D, TF, TI, Vector{TF}, Vector{TI}, Vector{Int}, Vector{Bool}}(
        to_HostVector(LBVH.enc),
        to_HostVector(LBVH.brt),
        to_HostVector(LBVH.leaf_aabb),
        to_HostVector(LBVH.node_aabb),
        Vector{TF}(LBVH.node_hmax),
        LBVH.root
    )
end

function PhantomRevealer.to_HostVector(grid :: GeneralGrid{D, TF, VG, VC}) where {D, TF <: AbstractFloat, VG <: CuVector{TF}, VC <: CuVector{NTuple{D, TF}}}
    return GeneralGrid{D, TF, Vector{TF}, Vector{NTuple{D, TF}}}(
        Vector{TF}(grid.grid),
        Vector{NTuple{D, TF}}(grid.coor)
    )
end

function PhantomRevealer.to_HostVector(grid :: StructuredGrid{D, TF, V, A}) where {D, TF <: AbstractFloat, V <: CuVector{TF}, A <: CuArray{TF, D}}
    return StructuredGrid{D, TF, Vector{TF}, Array{TF, D}}(
        Array{TF, D}(grid.grid),
        ntuple(i -> Vector{TF}(grid.axes[i]), D),
        grid.size
    )
end