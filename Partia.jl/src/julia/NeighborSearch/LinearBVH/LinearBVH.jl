"""
Linear Bounding Volume Hierarchy (LBVH) construction and traversal utilities  
    by Wei-Shan Su,  
    November 14, 2025
"""

################# Define structures #################
struct AABB{D, TF <: AbstractFloat, VF <: AbstractVector{TF}}
    min :: NTuple{D, VF}
    max :: NTuple{D, VF}
end

function Adapt.adapt_structure(to, x :: AB) where {D, AB <: AABB{D}}
    AABB(
        ntuple(i -> Adapt.adapt(to, x.min[i]), D),
        ntuple(i -> Adapt.adapt(to, x.max[i]), D)
    )
end

struct LinearBVH{D, TF <: AbstractFloat, VF <: AbstractVector{TF}, VB <: AbstractVector{Int32}}
    brt :: BinaryRadixTree{VB}
    leaf_coor :: NTuple{D, VF}
    leaf_h    :: VF
    node_aabb :: AABB{D, TF, VF}
    node_hmax :: VF
end

function Adapt.adapt_structure(to, x :: LBVH) where {D, LBVH <: LinearBVH{D}}
    LinearBVH(
        Adapt.adapt(to, x.brt),
        ntuple(i -> Adapt.adapt(to, x.leaf_coor[i]), D),
        Adapt.adapt(to, x.leaf_h),
        Adapt.adapt(to, x.node_aabb),
        Adapt.adapt(to, x.node_hmax),
    )
end

################# Constructing LBVH #################
"""
        LinearBVH(enc::MortonEncoding, brt::BinaryRadixTree)

Assemble a linear bounding volume hierarchy from a Morton-encoded particle set
and its matching binary radix tree. This stores per-leaf particle coordinates,
allocates per-node axis-aligned bounding boxes, discovers the tree root, and
precomputes the hierarchical extent data required for subsequent neighbor queries.

# Parameters
- `enc::MortonEncoding`: Morton-sorted particle coordinates and permutation.
- `brt::BinaryRadixTree`: Connectivity generated from the same `enc` instance.

# Returns
- `LinearBVH`: Immutable hierarchy storing the tree topology, leaf particle
    coordinates, and internal-node bounding volumes.
"""
function LinearBVH(enc::MortonEncoding{D, TF, TI, VF, VI}, brt::BinaryRadixTree{VB}) where {D, TF <: AbstractFloat, TI <: Unsigned, VF <: AbstractVector{TF}, VI <: AbstractVector{TI}, VB <: AbstractVector{Int32}}
    nleaf = brt.nleaf
    ninternal = nleaf - 1
    ntotal = 2 * nleaf - 1

    # (ninternal == 0) && return LBVH

    vproto = enc.coord[1]

    leaf_coor = ntuple(_ -> similar(vproto, nleaf), D)
    node_aabb = AABB(ntuple(_ -> similar(vproto, ninternal), D),
                     ntuple(_ -> similar(vproto, ninternal), D))
    node_hmax = similar(enc.h, ninternal)
    
    LBVH = LinearBVH{D, TF, VF, VB}(brt, leaf_coor, enc.h, node_aabb, node_hmax)

    visited = AtomicMemory{UInt32}(undef, ninternal)                        # atomic visit counters for internal nodes (2nd arrival processes the node)
    @threads for i in eachindex(visited)
        @inbounds @atomic :sequentially_consistent visited[i] = zero(UInt32)
    end

    _build_leaf_coords!(LBVH, enc)

    @threads for startid in Int32(nleaf):Int32(ntotal)                      # Karras: Each thread starts from one leaf node and walks up the tree using parent pointers that we record during radix tree construction.
        _build_internal_aabb!(LBVH, visited, startid)                       # Well I prefer to use id in 2n-1 space rather than the index of leaf
    end

    @inbounds for i in eachindex(visited)                                   # sanity: every internal node must be combined twice
        @assert (@atomic :sequentially_consistent visited[i]) >= UInt32(2)
    end

    return LBVH
end

function _build_leaf_coords!(LBVH::LinearBVH{D}, enc :: MortonEncoding{D}) where {D}
    coords = enc.coord
    leaf = LBVH.leaf_coor
    @inbounds for d in 1:D
        copyto!(leaf[d], coords[d])
    end
    return nothing
end

function _build_internal_aabb!(LBVH::LinearBVH{D}, visited :: AtomicMemory{UInt32}, startid :: Int32) where {D}
    brt = LBVH.brt
    nleaf = brt.nleaf

    # unified parent: length = 2nleaf-1, parent[root]=0, parent[leaf/internal]=parent internal id
    parent = brt.parent

    # AABB buffers
    node_min  = LBVH.node_aabb.min
    node_max  = LBVH.node_aabb.max
    node_hmax = LBVH.node_hmax
    leaf      = LBVH.leaf_coor
    leaf_h    = LBVH.leaf_h

    # BRT children (only meaningful for internal ids 1..ninternal)
    left  = brt.left
    right = brt.right

    # climb starts from parent(startid); startid is a leaf unified node id in nleaf..2nleaf-1
    p = @inbounds parent[Int(startid)]   # internal id (1..ninternal) or 0

    while p != 0
        pidx = internal_index(p)

        # Karras: combine on the second arrival (newval == 2)
        newval = @atomic :sequentially_consistent visited[pidx] += one(UInt32)

        if newval == UInt32(2)
            # second arrival: both children are ready => combine
            @inbounds begin
                l = left[pidx]
                r = right[pidx]

                # hmax
                hl = is_leaf_id(l, nleaf) ? leaf_h[leaf_index(l, nleaf)] : node_hmax[internal_index(l)]
                hr = is_leaf_id(r, nleaf) ? leaf_h[leaf_index(r, nleaf)] : node_hmax[internal_index(r)]
                node_hmax[pidx] = ifelse(hl > hr, hl, hr)

                # bounds
                for d in 1:D
                    lmin = is_leaf_id(l, nleaf) ? leaf[d][leaf_index(l, nleaf)] : node_min[d][internal_index(l)]
                    rmin = is_leaf_id(r, nleaf) ? leaf[d][leaf_index(r, nleaf)] : node_min[d][internal_index(r)]
                    lmax = is_leaf_id(l, nleaf) ? leaf[d][leaf_index(l, nleaf)] : node_max[d][internal_index(l)]
                    rmax = is_leaf_id(r, nleaf) ? leaf[d][leaf_index(r, nleaf)] : node_max[d][internal_index(r)]
                    node_min[d][pidx] = ifelse(lmin < rmin, lmin, rmin)
                    node_max[d][pidx] = ifelse(lmax > rmax, lmax, rmax)
                end
            end

            # climb to parent of this internal node id (internal ids are valid indices in parent[])
            p = @inbounds parent[pidx]
        else
            # first arrival
            break
        end
    end

    return nothing
end