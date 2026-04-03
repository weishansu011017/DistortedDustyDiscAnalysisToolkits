"""
Binary Radix Tree construction and data structure implementation  
    by Wei-Shan Su,  
    November 7, 2025
"""

################# Define structures #################
struct BinaryRadixTree{V<:AbstractVector{Int32}}
    root   :: Int32   
    nleaf  :: Int
    left   :: V                     # length = 2*nleaf-1
    right  :: V                     # length = 2*nleaf-1
    escape :: V                     # length = 2*nleaf-1
    parent :: V                     # length = 2*nleaf-1
end

function Adapt.adapt_structure(to, x :: BinaryRadixTree)
    BinaryRadixTree(
        x.root,
        x.nleaf,
        Adapt.adapt(to, x.left),
        Adapt.adapt(to, x.right),
        Adapt.adapt(to, x.escape),
        Adapt.adapt(to, x.parent)
    )
end

################# Constructing Binary Radix Tree #################
"""
    BinaryRadixTree(enc::MortonEncoding)

Construct a **Binary Radix Tree (BRT)** from a Morton-sorted code array `enc.codes`,
using the Karras (2012) range/split construction and a **stackless escape table**
(Prokopenko & Lebrun-Grandié, 2024) for traversal without parent-walking.

The constructor builds the tree connectivity in *linear time* from the sorted Morton
codes. Children are stored as **unified node IDs** in a single node-id space of size
`2n-1` (A1 layout), where `n = length(enc.codes)`:

- internal node IDs: `1:(n-1)`
- leaf node IDs: `n:(2n-1)`  (leaf index `k ∈ 1:n` maps to node ID `(n-1) + k`)
- `0` is a sentinel meaning “no node” / termination

In addition, `escape` is computed for **all unified node IDs** (both internal and leaf).
During stackless traversal, `escape[id]` gives the next node to visit after finishing
(or pruning) the subtree rooted at `id`. Escape targets are derived from the adjacent
longest-common-prefix-length array `adj`, with a `+∞` sentinel at `adj[n]`.

This implementation also stores a **unified parent array** `parent` (length `2n-1`,
`parent[root]=0`). It is not required for stackless traversal when `escape` is used,
but is useful for bottom-up refit (Karras-style atomic “second arrival” reduction).

# Parameters
- `enc :: MortonEncoding{D, TF, TI, VF, VI}`
  Container holding the sorted Morton codes (`enc.codes`).

# Returns
- `BinaryRadixTree{Vector{Int32}}`
  A struct containing:
  - `root`   — root unified node ID (`1` if `n ≥ 2`, otherwise `0`)
  - `nleaf`  — number of leaves `n`
  - `left`   — left child node ID for each node (length `2n-1`; meaningful for internal nodes)
  - `right`  — right child node ID for each node (length `2n-1`; meaningful for internal nodes)
  - `escape` — escape node ID for each node (length `2n-1`; defined for both internal and leaf nodes)
  - `parent` — parent internal ID for each unified node ID (length `2n-1`; `0` for the root)

# Notes
- `enc.codes` must be sorted; otherwise the constructed tree and escape links are invalid.
- For leaf nodes, `left` and `right` are unused (typically left as `0`).
- This implementation stores node IDs in `Int32`. Therefore the maximum supported leaf count is
  `n ≤ 1_073_741_824` (since `2n-1` must fit in signed 32-bit). 

# Reference
- Karras, T. (2012). *Maximizing Parallelism in the Construction of BVHs, Octrees, and k-d Trees*.  
    In *High Performance Graphics 2012* (pp. 33–37).  
    DOI: [10.2312/EGGH/HPG12/033-037]
- Prokopenko, A., & Lebrun-Grandié, D. (2024). *Revising Apetrei's bounding volume hierarchy
    construction algorithm to allow stackless traversal*. Oak Ridge National Laboratory
    Technical Report. DOI: 10.2172/2301619
"""
function BinaryRadixTree(enc::MortonEncoding{D, TF, TI, VF, VI}) where {D, TF <: AbstractFloat, TI <: Unsigned, VF <: AbstractVector{TF}, VI <: AbstractVector{TI}}
    # Properties of BRT
    codes = enc.codes
    n = length(codes)                    # Int
    n >= 1 || throw(ArgumentError("BinaryRadixTree: enc.codes must be non-empty (got n=0)."))
    root = (n >= 2) ? one(Int32) : zero(Int32)

    # Int32 node-id capacity: total_length = 2n-1 must fit in Int32
    n_max = (typemax(Int32) ÷ 2) + 1  # 1_073_741_824
    n <= n_max || throw(ArgumentError("BinaryRadixTree: n=$n exceeds the Int32 node-id capacity (requires 2n-1 ≤ typemax(Int32); n ≤ $n_max)."))

    n_internal = n - 1                   # Int
    total_length = 2*n - 1               # Int

    # Initializing arrays (length = 2n - 1)
    left   = zeros(Int32, total_length)
    right  = zeros(Int32, total_length)
    escape = zeros(Int32, total_length)
    parent = zeros(Int32, total_length)

    # range right (length = ninternal)
    range_hi = Vector{Int32}(undef, n_internal)

    if n_internal > 0
        # Loop: establishing the Karras BRT
        @threads for i in 1:n_internal
            _build_child!(left, right, range_hi, parent, codes, i)
        end

        # Adjacent longest-common-prefix lengths (Algorithm 2, Prokopenko & Lebrun-Grandié 2024)
        adj = Vector{Int32}(undef, n)
        @threads for i in 1:n-1
            _build_adjacent!(adj, codes, i)
        end
        adj[n] = typemax(Int32)   # +∞ sentinel for the last slot

        # Internal-node escapes (Algorithm 2)
        @threads for i in 1:n_internal
            _build_escape!(escape, adj, range_hi, n, i)
        end

        # Leaf escapes are derived from parent + sibling relationships
        for leaf_id in n:total_length
            _build_escape!(escape, parent, left, right, leaf_id)
        end
    else
        escape[1] = 0
        parent[1] = 0
    end
    
    return BinaryRadixTree{Vector{Int32}}(root, n, left, right, escape, parent)
end

# Toolbox
@inline is_leaf_id(node::Int32, nleaf::Int) = (node != 0) & (node >= Int32(nleaf))
@inline is_internal_id(node::Int32, nleaf::Int) = (node != 0) & (node < Int32(nleaf))
@inline leaf_index(node::Int32, nleaf::Int) = Int(node) - (nleaf - 1)               # 1..nleaf
@inline internal_index(node::Int32) = Int(node)                                     # 1..ninternal

@inline function _range_direction(codes::V, i::Int) where {TI<:Unsigned, V<:AbstractVector{TI}}
    δL = (i > 1) ?  _longest_common_prefix_length(codes, i, i - 1) : -1
    δR = (i < length(codes)) ? _longest_common_prefix_length(codes, i, i + 1) : -1
    d = sign(δR - δL)
    return (d == 0) ? 1 : d
end


@inline function _find_range(codes :: V, i :: Int) where {TI <: Unsigned, V<:AbstractVector{TI}}
    n = length(codes)
    d = _range_direction(codes, i)
    δmin = (1 <= i - d <= length(codes)) ? _longest_common_prefix_length(codes, i, i - d) : -1

    # Find the upper limit
    lmax = 2
    while true
        j = i + lmax * d
        if (j < 1) || (j > n)
            break
        end
        δtest = _longest_common_prefix_length(codes, i, j)
        if δtest <= δmin
            break
        end
        lmax *= 2
    end
    # Find the other side
    l = 0
    t = lmax >> 1    # lmax / 2, still Int
    while t > 0
        j = i + (l + t) * d
        if j >= 1 && j <= n && _longest_common_prefix_length(codes, i, j) > δmin
            l += t
        end
        t >>= 1      # t = t / 2, still Int
    end

    j = i + l * d
    return (min(i, j), max(i, j))  
end

@inline function _split_position(codes::V, first::Int, last::Int) where {TI<:Unsigned, V<:AbstractVector{TI}}
    if codes[first] == codes[last]
        return clamp((first + last) >> 1, first, last - 1)
    end

    @inbounds prefix_first_last = _longest_common_prefix_length(codes, first, last)
    split = first
    step = last - first

    while step > 1
        step = (step + 1) >> 1
        new_split = split + step
        if new_split < last
            prefix = _longest_common_prefix_length(codes, first, new_split)
            if prefix > prefix_first_last
                split = new_split
            end
        end
    end

    return split
end

@inline function _build_adjacent!(adj :: S, codes :: V, i :: Int) where {TI<:Unsigned, V<:AbstractVector{TI},  S<:AbstractVector{Int32}}
    adj[i] = _longest_common_prefix_length(codes, i)
    return nothing                  
end

@inline function _build_child!(left :: S, right :: S, range_hi :: S, parent :: S, codes :: V, i :: Int) where {TI<:Unsigned, V<:AbstractVector{TI},  S<:AbstractVector{Int32}}
    n = length(codes)
    leaf_offset = n - 1
    
    @inbounds begin
        lo, hi = _find_range(codes, i)
        range_hi[i] = Int32(hi)
        s = _split_position(codes, lo, hi)

        # NOTE (threaded build): We assume each child node has a unique parent in a valid BRT.
        # If enc.codes is not sorted, or duplicate-code tie-breaking is broken, the topology may
        # become invalid and the same `idx` could be assigned by multiple `i` concurrently,
        # causing nondeterministic overwrites in `parent[idx]`.

        # left child
        if s == lo
            idx = lo + leaf_offset              # leaf node id in n..2n-1
        else
            idx = s                             # internal node id in 1..n-1
        end
        left[i] = Int32(idx)
        parent[idx] = Int32(i)

        # right child
        if s + 1 == hi
            idx = hi + leaf_offset
        else
            idx = s + 1
        end
        right[i] = Int32(idx)
        parent[idx] = Int32(i)
    end

    return nothing
end

@inline function _build_escape!(escape::V, adj::V, range_hi::V, nleaf::Int,i::Int) where {V<:AbstractVector{Int32}}
    # Escape for internal node
    leaf_offset = nleaf - 1
    @inbounds begin
        hi = Int(range_hi[i])   # hi is leaf index in 1..nleaf
        if hi == nleaf
            escape[i] = 0
        elseif hi == nleaf - 1
            escape[i] = Int32(leaf_offset + nleaf)
        else
            next = hi + 1
            escape[i] = (adj[next] < adj[next-1]) ? Int32(leaf_offset + next) : Int32(next)
        end
    end
    return nothing
end

@inline function _build_escape!(escape::V, parent::V, left::V, right::V, i::Int) where {V<:AbstractVector{Int32}}
    @inbounds begin
        p = parent[i]              # unified internal id (1..nleaf-1) or 0
        if iszero(p)
            escape[i] = 0
            return nothing
        end

        pidx = internal_index(p)         # 1..ninternal
        if left[pidx] == Int32(i)
            # next is the sibling subtree root
            escape[i] = right[pidx]
        else
            # must be right child => go where parent would escape to
            # p is unified id, so escape[p] is valid
            escape[i] = escape[Int(p)]
        end
    end
    return nothing
end

function _build_escape_stackless!(escape::V, left::V, right::V, parent::V, root::Int32, nleaf::Int) where {V<:AbstractVector{Int32}}
    ntotal = 2nleaf - 1
    @assert length(escape) == ntotal
    @assert length(left) == ntotal
    @assert length(right) == ntotal
    @assert length(parent) == ntotal

    fill!(escape, Int32(0))
    root == 0 && return nothing

    stack = Int32[root]
    while !isempty(stack)
        node = pop!(stack)
        is_internal_id(node, nleaf) || continue

        node_idx = internal_index(node)
        l = left[node_idx]
        r = right[node_idx]

        # left subtree escapes to its sibling; right subtree inherits parent's escape
        escape[Int(l)] = r
        escape[Int(r)] = escape[Int(node)]

        # process children; push right first so left is popped/visited first (preorder)
        push!(stack, r)
        push!(stack, l)
    end

    return nothing
end



