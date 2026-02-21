@inline function _balance!(M::MMatrix{N,N,ComplexF64}, bal::MVector{N,Float64}) where {N}
    n = LinearAlgebra.BlasInt(N)
    lda = LinearAlgebra.BlasInt(max(1, stride(M,2)))
    ilo = Ref{LinearAlgebra.BlasInt}(); ihi = Ref{LinearAlgebra.BlasInt}(); info = Ref{LinearAlgebra.BlasInt}(0)
    GC.@preserve M bal begin
        ccall((:zgebal_64_, LinearAlgebra.LAPACK.liblapack), Cvoid,
              (Ref{UInt8}, Ref{LinearAlgebra.BlasInt}, Ptr{ComplexF64}, Ref{LinearAlgebra.BlasInt}, Ref{LinearAlgebra.BlasInt}, Ref{LinearAlgebra.BlasInt}, Ptr{Float64}, Ref{LinearAlgebra.BlasInt}),
              Ref(UInt8('B')), Ref(n), pointer(M), Ref(lda), ilo, ihi, pointer(bal), info)
    end
    info[] == 0 || error("zgebal failed, info=$(info[])")
    return Int(ilo[]), Int(ihi[])
end

@inline function _balance!(M::MMatrix{N,N,ComplexF32}, bal::MVector{N,Float32}) where {N}
    n = LinearAlgebra.BlasInt(N)
    lda = LinearAlgebra.BlasInt(max(1, stride(M,2)))
    ilo = Ref{LinearAlgebra.BlasInt}(); ihi = Ref{LinearAlgebra.BlasInt}(); info = Ref{LinearAlgebra.BlasInt}(0)
    GC.@preserve M bal begin
        ccall((:cgebal_64_, LinearAlgebra.LAPACK.liblapack), Cvoid,
              (Ref{UInt8}, Ref{LinearAlgebra.BlasInt}, Ptr{ComplexF32}, Ref{LinearAlgebra.BlasInt}, Ref{LinearAlgebra.BlasInt}, Ref{LinearAlgebra.BlasInt}, Ptr{Float32}, Ref{LinearAlgebra.BlasInt}),
              Ref(UInt8('B')), Ref(n), pointer(M), Ref(lda), ilo, ihi, pointer(bal), info)
    end
    info[] == 0 || error("cgebal failed, info=$(info[])")
    return Int(ilo[]), Int(ihi[])
end