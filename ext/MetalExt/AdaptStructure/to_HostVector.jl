

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
    )
end

function PhantomRevealer.to_HostVector(brt :: BinaryRadixTree{TI, VI, A, B}) where {TI <: Unsigned, VI <: MtlVector{TI},  A <: MtlVector{Int}, B <: MtlVector{Bool}}
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

function PhantomRevealer.to_HostVector(AB :: AABB{D, TF, VF}) where {D, TF <: Float32, VF <: MtlVector{TF}}
    return AABB{D, Float32, Vector{Float32}}(
        ntuple(i -> Vector{Float32}(AB.min[i]),D),
        ntuple(i -> Vector{Float32}(AB.max[i]),D)
    )

end

function PhantomRevealer.to_HostVector(LBVH :: LinearBVH{D, TF, TI, VF, VI, A, B}) where {D, TF <: Float32, TI <: Unsigned, VF <: MtlVector{TF}, VI <: MtlVector{TI}, A <: MtlVector{Int}, B <: MtlVector{Bool}}
    return LinearBVH{D, Float32, TI, Vector{Float32}, Vector{TI}, Vector{Int}, Vector{Bool}}(
        to_HostVector(LBVH.enc),
        to_HostVector(LBVH.brt),
        to_HostVector(LBVH.leaf_aabb),
        to_HostVector(LBVH.node_aabb),
        Vector{Float32}(LBVH.node_hmax),
        LBVH.root
    )
end

function PhantomRevealer.to_HostVector(grid :: GeneralGrid{D, TF, VG, VC}) where {D, TF <: Float32, VG <: MtlVector{TF}, VC <: MtlVector{NTuple{D, TF}}}
    return GeneralGrid{D, Float32, Vector{Float32}, Vector{NTuple{D, Float32}}}(
        Vector{Float32}(grid.grid),
        Vector{NTuple{D, Float32}}(grid.coor)
    )
end

function PhantomRevealer.to_HostVector(grid :: StructuredGrid{D, TF, V, A}) where {D, TF <: Float32, V <: MtlVector{TF}, A <: MtlArray{TF, D}}
    return StructuredGrid{D, Float32, Vector{Float32}, Array{Float32, D}}(
        Array{Float32, D}(grid.grid),
        ntuple(i -> Vector{Float32}(grid.axes[i]), D),
        grid.size
    )
end