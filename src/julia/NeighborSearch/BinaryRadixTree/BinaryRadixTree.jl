
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

@inline function _split_position(codes::V, i::Int, j::Int) where {TI<:Unsigned, V <: AbstractVector{TI}}
    @inbounds δnode = _longest_common_prefix_length(codes[i], codes[j])
    lo = i
    hi = j - 1
    while lo < hi
        mid = (lo + hi) >>> 1
        if _longest_common_prefix_length(codes[i], codes[mid]) > δnode
            lo = mid + 1
        else
            hi = mid
        end
    end
    
    return max(i, lo - 1)
end

@inline function _find_range(codes :: V, i :: Int) where {TI <: Unsigned, V<:AbstractVector{TI}}
    n = length(codes)
    d = _range_direction(codes, i)
    δmin = (1 <= i - d <= length(codes)) ? _longest_common_prefix_length(codes[i], codes[i - d]) : -1

    # Find the upper limit
    lmax = 2
    while true
        j = i + lmax * d
        if (j < 1) || (j > n - 1)
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

@inline function _build_binary_radix_tree!(left_child :: S, right_child :: S, is_leaf_left :: S, is_leaf_right :: S, codes::V, i :: Int) where {TI<:Unsigned, V<:AbstractVector{TI},  S<:AbstractVector{TI}}
    @inbounds begin
        lo, hi = _find_range(codes, i)
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

function build_binary_radix_tree(enc::MortonEncoding{D, TF, TI, V}) where {D, TF <: AbstractFloat, TI <: Unsigned, V <: AbstractVector{TI}}
    codes = enc.codes
    n = length(codes)
    n_internal = n - 1

    left_child    = similar(codes, n_internal)
    right_child   = similar(codes, n_internal)
    is_leaf_left  = zeros(Bool, n_internal)
    is_leaf_right = zeros(Bool, n_internal)

    @threads for i in eachindex(left_child, right_child, is_leaf_left, is_leaf_right)
        _build_binary_radix_tree!(left_child, right_child, is_leaf_left, is_leaf_right, codes, i)
    end
    return BinaryRadixTree(left_child, right_child, is_leaf_left, is_leaf_right)
end

