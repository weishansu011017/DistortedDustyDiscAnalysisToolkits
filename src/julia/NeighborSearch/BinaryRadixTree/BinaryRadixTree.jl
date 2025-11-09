"""
Binary Radix Tree construction and data structure implementation  
    by Wei-Shan Su,  
    November 7, 2025
"""

struct BinaryRadixTree{TI <: Unsigned, VI <: AbstractVector{TI},  A <: AbstractVector{Int}, B <: AbstractVector{Bool}}
    left_child    :: A
    right_child   :: A
    is_leaf_left  :: B
    is_leaf_right :: B

    # For GPU LBVH construction
    leaf_parent   :: VI
    node_parent   :: VI
    visit_counter :: VI
end


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
    t = lmax >> 1    # lmax / 2
    while t > 0
        j = i + (l + t) * d
        if j >= 1 && j <= n && _longest_common_prefix_length(codes, i, j) > δmin
            l += t
        end
        t >>= 1      # t = t / 2
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

@inline function _build_binary_radix_tree!(left_child :: S, right_child :: S, is_leaf_left :: U, is_leaf_right :: U, codes::V, i :: Int) where {TI<:Unsigned, V<:AbstractVector{TI},  S<:AbstractVector{Int}, U<:AbstractVector{Bool}}
    @inbounds begin
        lo, hi = _find_range(codes, i)
        if lo == hi
            left_child[i]  = lo
            right_child[i] = lo
            is_leaf_left[i] = true
            is_leaf_right[i] = true
            return
        end
        s = _split_position(codes, lo, hi)

        if s == lo
            left_child[i] = lo
            is_leaf_left[i] = true
        else
            left_child[i] = s
            is_leaf_left[i] = false
        end

        if s + 1 == hi
            right_child[i] = hi
            is_leaf_right[i] = true
        else
            right_child[i] = s + 1
            is_leaf_right[i] = false
        end
    end
end

@inline function _build_parent!(left_child :: S, right_child :: S, is_leaf_left :: U, is_leaf_right :: U, node_parent :: X, leaf_parent :: W) where {TI<:Unsigned, S<:AbstractVector{Int}, U<:AbstractVector{Bool}, X<:AbstractVector{TI}, W<:AbstractVector{TI}}
    @inbounds for i in eachindex(left_child)
        l = left_child[i]
        r = right_child[i]

        if is_leaf_left[i]
            leaf_parent[l] = i
        else
            node_parent[l] = i
        end

        if is_leaf_right[i]
            leaf_parent[r] = i
        else
            node_parent[r] = i
        end
    end
    return nothing
end

"""
    BinaryRadixTree(enc::MortonEncoding)

Constructs a **Binary Radix Tree (BRT)** from a sorted sequence of Morton codes.

This function implements the *linear-time* construction algorithm described in  
Karras (2012), which builds the hierarchical connectivity of a binary radix tree  
directly from a pre-sorted Morton code array. Each internal node encodes its  
two child indices and flags indicating whether each child is a leaf or an internal node.

# Parameters
- `enc :: MortonEncoding{D, TF, TI, VF, VI}`  
  A Morton encoding container holding the sorted Morton codes (`enc.codes`) and  
  spatial coordinate arrays.

# Returns
- `BinaryRadixTree{TI, VI, Vector{Int}, Vector{Bool}}`  
  A struct containing:
  - `left_child`    — index of the left child for each internal node  
  - `right_child`   — index of the right child for each internal node  
  - `is_leaf_left`  — whether the left child is a leaf node  
  - `is_leaf_right` — whether the right child is a leaf node  
  - `leaf_parent`   — parent index for each leaf node  
  - `node_parent`   — parent index for each internal node  
  - `visit_counter` — counter used for tree traversal bookkeeping

# Reference
Karras, T. (2012). *Maximizing Parallelism in the Construction of BVHs, Octrees, and k-d Trees*.  
In *High Performance Graphics 2012* (pp. 33–37).  
DOI: [10.2312/EGGH/HPG12/033-037]
"""
function BinaryRadixTree(enc::MortonEncoding{D, TF, TI, VF, VI}) where {D, TF <: AbstractFloat, TI <: Unsigned, VF <: AbstractVector{TF}, VI <: AbstractVector{TI}}
    codes = enc.codes
    n = length(codes)
    n_internal = n - 1

    left_child    = zeros(Int, n_internal)
    right_child   = zeros(Int, n_internal)
    is_leaf_left  = zeros(Bool, n_internal)
    is_leaf_right = zeros(Bool, n_internal)

    leaf_parent   = similar(codes, TI, n)
    node_parent   = similar(codes, TI, n_internal)
    visit_counter = similar(codes, TI, n_internal)

    fill!(leaf_parent, zero(TI))
    fill!(node_parent, zero(TI))
    fill!(visit_counter, zero(TI))


    @threads for i in eachindex(left_child, right_child, is_leaf_left, is_leaf_right)
        _build_binary_radix_tree!(left_child, right_child, is_leaf_left, is_leaf_right, codes, i)
    end
    _build_parent!(left_child, right_child, is_leaf_left, is_leaf_right, node_parent, leaf_parent)
    return BinaryRadixTree{TI, VI, typeof(left_child), typeof(is_leaf_left)}(left_child, right_child, is_leaf_left, is_leaf_right,leaf_parent , node_parent, visit_counter)
end

