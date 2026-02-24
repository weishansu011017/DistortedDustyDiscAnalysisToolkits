@inline function _scale!(M :: MMatrix{N, N, T}) where {N, T <: Complex}
    RT = Base._realtype(T)

    # Find the maximum absolute value in M i.e. max_ij ||A||_ij
    maxnum = maxabs(M)

    # Define the small number & large number
    smlnum  = _smlnum(RT)
    bignum  = _bignum(RT)

    # Comparison
    scale = one(RT)
    if isfinite(maxnum) && maxnum > 0
        if maxnum < smlnum
            scale = smlnum / maxnum
        elseif maxnum > bignum
            scale = bignum / maxnum
        end
    end

    # Apply to the matrix
    @inbounds for i in eachindex(M)
        M[i] *= scale
    end

    return scale
end

@inline function _unscale!(W::MVector{N,T}, α) where {N,T<:Complex}
    @inbounds for i in 1:N
        W[i] /= α
    end
    return nothing
end

# Toolbox
@inline _smlnum(::Type{T}) where {T<:AbstractFloat} = floatmin(T) / eps(T)
@inline _bignum(::Type{T}) where {T<:AbstractFloat} = inv(_smlnum(T))