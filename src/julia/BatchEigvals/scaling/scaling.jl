@inline _realtype(::Type{Complex{TF}}) where {TF <: AbstractFloat} = TF
@inline _realtype(::Type{T}) where {T} = throw(ArgumentError("Unsupported element type $T; Expected Complex{<:AbstractFloat}"))
@inline function _scale!(M::MMatrix{N,N,ComplexF64}) where {N}
    anrm = c_zlange_M(M)
    eps = c_dlamch(UInt8('P'))
    smlnum = c_dlamch(UInt8('S'))
    bignum = 1 / smlnum
    smlnum = sqrt(smlnum) / eps
    bignum = 1 / smlnum
    scaleA = false
    cscale = 1.0
    if anrm > 0 && anrm < smlnum
        cscale = smlnum; scaleA = true
    elseif anrm > bignum
        cscale = bignum; scaleA = true
    end
    if scaleA
        info = c_zlascl_G!(anrm, cscale, M)
        info == 0 || error("zlascl failed, info=$(info)")
        return true, cscale / anrm
    else
        return false, 1.0
    end
end

@inline function _scale!(M::MMatrix{N,N,ComplexF32}) where {N}
    anrm = c_clange_M(M)
    eps = c_slamch(UInt8('P'))
    smlnum = c_slamch(UInt8('S'))
    bignum = 1f0 / smlnum
    smlnum = sqrt(smlnum) / eps
    bignum = 1f0 / smlnum
    scaleA = false
    cscale = 1f0
    if anrm > 0f0 && anrm < smlnum
        cscale = smlnum; scaleA = true
    elseif anrm > bignum
        cscale = bignum; scaleA = true
    end
    if scaleA
        info = c_clascl_G!(anrm, cscale, M)
        info == 0 || error("clascl failed, info=$(info)")
        return true, cscale / anrm
    else
        return false, 1f0
    end
end

@inline function _unscale!(W::MVector{N,T}, α) where {N,T<:Complex}
    invα = inv(α)
    @inbounds for i in 1:N
        W[i] *= invα
    end
    return nothing
end

# Toolbox
@inline c_dlamch(c::UInt8) = ccall((:dlamch_64_, LinearAlgebra.LAPACK.liblapack), Float64, (Ref{UInt8},), Ref(c))
@inline c_slamch(c::UInt8) = ccall((:slamch_64_, LinearAlgebra.LAPACK.liblapack), Float32, (Ref{UInt8},), Ref(c))

@inline function c_zlange_M(A::MMatrix{N,N,ComplexF64}) where {N}
    m = LinearAlgebra.BlasInt(N); n = LinearAlgebra.BlasInt(N); lda = LinearAlgebra.BlasInt(max(1, stride(A,2)))
    work = Vector{Float64}(undef, max(1, N))
    GC.@preserve A work begin
        return ccall((:zlange_64_, LinearAlgebra.LAPACK.liblapack), Float64,
                 (Ref{UInt8}, Ref{LinearAlgebra.BlasInt}, Ref{LinearAlgebra.BlasInt}, Ptr{ComplexF64}, Ref{LinearAlgebra.BlasInt}, Ptr{Float64}),
                     Ref(UInt8('M')), Ref(m), Ref(n), pointer(A), Ref(lda), pointer(work))
    end
end

@inline function c_clange_M(A::MMatrix{N,N,ComplexF32}) where {N}
    m = LinearAlgebra.BlasInt(N); n = LinearAlgebra.BlasInt(N); lda = LinearAlgebra.BlasInt(max(1, stride(A,2)))
    work = Vector{Float32}(undef, max(1, N))
    GC.@preserve A work begin
        return ccall((:clange_64_, LinearAlgebra.LAPACK.liblapack), Float32,
                 (Ref{UInt8}, Ref{LinearAlgebra.BlasInt}, Ref{LinearAlgebra.BlasInt}, Ptr{ComplexF32}, Ref{LinearAlgebra.BlasInt}, Ptr{Float32}),
                     Ref(UInt8('M')), Ref(m), Ref(n), pointer(A), Ref(lda), pointer(work))
    end
end

@inline function c_zlascl_G!(cfrom::Float64, cto::Float64, A::MMatrix{N,N,ComplexF64}) where {N}
    typ = UInt8('G'); kl = LinearAlgebra.BlasInt(0); ku = LinearAlgebra.BlasInt(0)
    m = LinearAlgebra.BlasInt(N); n = LinearAlgebra.BlasInt(N); lda = LinearAlgebra.BlasInt(max(1, stride(A,2)))
    info = Ref{LinearAlgebra.BlasInt}(0)
    GC.@preserve A begin
          ccall((:zlascl_64_, LinearAlgebra.LAPACK.liblapack), Cvoid,
              (Ref{UInt8}, Ref{LinearAlgebra.BlasInt}, Ref{LinearAlgebra.BlasInt}, Ref{Float64}, Ref{Float64}, Ref{LinearAlgebra.BlasInt}, Ref{LinearAlgebra.BlasInt}, Ptr{ComplexF64}, Ref{LinearAlgebra.BlasInt}, Ref{LinearAlgebra.BlasInt}),
              Ref(typ), Ref(kl), Ref(ku), Ref(cfrom), Ref(cto), Ref(m), Ref(n), pointer(A), Ref(lda), info)
    end
    return info[]
end

@inline function c_clascl_G!(cfrom::Float32, cto::Float32, A::MMatrix{N,N,ComplexF32}) where {N}
    typ = UInt8('G'); kl = LinearAlgebra.BlasInt(0); ku = LinearAlgebra.BlasInt(0)
    m = LinearAlgebra.BlasInt(N); n = LinearAlgebra.BlasInt(N); lda = LinearAlgebra.BlasInt(max(1, stride(A,2)))
    info = Ref{LinearAlgebra.BlasInt}(0)
    GC.@preserve A begin
          ccall((:clascl_64_, LinearAlgebra.LAPACK.liblapack), Cvoid,
              (Ref{UInt8}, Ref{LinearAlgebra.BlasInt}, Ref{LinearAlgebra.BlasInt}, Ref{Float32}, Ref{Float32}, Ref{LinearAlgebra.BlasInt}, Ref{LinearAlgebra.BlasInt}, Ptr{ComplexF32}, Ref{LinearAlgebra.BlasInt}, Ref{LinearAlgebra.BlasInt}),
              Ref(typ), Ref(kl), Ref(ku), Ref(cfrom), Ref(cto), Ref(m), Ref(n), pointer(A), Ref(lda), info)
    end
    return info[]
end