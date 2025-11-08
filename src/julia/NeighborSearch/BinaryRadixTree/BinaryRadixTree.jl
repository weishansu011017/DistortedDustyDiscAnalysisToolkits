"""
Binary Radix Tree construction and data structure implementation  
    by Wei-Shan Su,  
    November 7, 2025
"""

struct BinaryRadixTree{A <: AbstractVector{Int}, B <: AbstractVector{Bool}}
    left_child :: A
    right_child :: A
    is_leaf_left :: B
    is_leaf_right :: B
end


@inline function _range_direction(codes::V, i::Int) where {TI <: Unsigned, V<:AbstractVector{TI}}
    δL = (i > 1) ? _longest_common_prefix_length(codes[i], codes[i-1]) : -1
    δR = (i < length(codes)) ? _longest_common_prefix_length(codes[i], codes[i+1]) : -1
    d = sign(δR - δL)
    return (d == 0) ? 1 : d
end

@inline function _find_range(codes :: V, i :: Int) where {TI <: Unsigned, V<:AbstractVector{TI}}
    n = length(codes)
    d = _range_direction(codes, i)
    δmin = (1 <= i - d <= length(codes)) ? _longest_common_prefix_length(codes[i], codes[i - d]) : -1

    # If all of them is identical
    code_i = codes[i]
    lo = i
    hi = i
    while (lo > 1) && (codes[lo - 1] == code_i)
        lo -= 1
    end
    while (hi < n) && (codes[hi + 1] == code_i)
        hi += 1
    end
    if (lo != i) || (hi != i)
        return (lo, hi)
    end

    # Find the upper limit
    lmax = 2
    while true
        j = i + lmax * d
        if (j < 1) || (j > n)
            break
        end
        δtest = _longest_common_prefix_length(codes[i], codes[j])
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
        if j >= 1 && j <= n && _longest_common_prefix_length(codes[i], codes[j]) > δmin
            l += t
        end
        t >>= 1      # t = t / 2
    end

    j = i + l * d
    return (min(i, j), max(i, j))  
end

@inline function _split_position(codes::V, i::Int, j::Int) where {TI<:Unsigned, V <: AbstractVector{TI}}
    if codes[i] == codes[j]    
        return (i + j) >>> 1
    end
    @inbounds δnode = _longest_common_prefix_length(codes[i], codes[j])
    lo = i
    hi = j - 1
    while lo < hi
        mid = (lo + hi) >>> 1
        if _longest_common_prefix_length(codes[i], codes[mid]) >= δnode
            lo = mid + 1
        else
            hi = mid
        end
    end
    
    return max(i, lo - 1)
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

"""
    BinaryRadixTree(enc::MortonEncoding{D, TF, TI, V})

Construct a Binary Radix Tree (BRT) from a sorted Morton encoding sequence.

This function implements the linear-time construction algorithm proposed by Karras (2012),
which builds the internal node connectivity of a radix tree from a pre-sorted array of
Morton codes. Each internal node stores indices of its left and right children,
and whether each child is a leaf or another internal node.

# Parameters
- `enc :: MortonEncoding{D, TF, TI, V}` :  
  A Morton encoding container that holds the sorted Morton codes (`enc.codes`).

# Returns
- `BinaryRadixTree{Vector{Int}, Vector{Bool}}` :  
  A struct containing:
  - `left_child`  – index of the left child for each internal node  
  - `right_child` – index of the right child for each internal node  
  - `is_leaf_left`  – whether the left child is a leaf node  
  - `is_leaf_right` – whether the right child is a leaf node

# Reference
- Karras, T. (2012). *Maximizing Parallelism in the Construction of BVHs, Octrees, and k-d Trees*.  
  Proceedings of High Performance Graphics 2012. doi:[10.2312/EGGH/HPG12/033-037]
"""
function BinaryRadixTree(enc::MortonEncoding{D, TF, TI, V}) where {D, TF <: AbstractFloat, TI <: Unsigned, V <: AbstractVector{TI}}
    codes = enc.codes
    n = length(codes)
    n_internal = n - 1

    left_child    = zeros(Int, n_internal)
    right_child   = zeros(Int, n_internal)
    is_leaf_left  = zeros(Bool, n_internal)
    is_leaf_right = zeros(Bool, n_internal)

    @threads for i in eachindex(left_child, right_child, is_leaf_left, is_leaf_right)
        _build_binary_radix_tree!(left_child, right_child, is_leaf_left, is_leaf_right, codes, i)
    end
    return BinaryRadixTree(left_child, right_child, is_leaf_left, is_leaf_right)
end

