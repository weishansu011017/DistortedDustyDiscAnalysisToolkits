@inline function _schur_eigvals!(M::MMatrix{N,N,ComplexF64}, ilo::Int, ihi::Int, W::MVector{N,ComplexF64}, work::MVector{L,ComplexF64}) where {N,L}
    n = LinearAlgebra.BlasInt(N)
    ldh = LinearAlgebra.BlasInt(max(1, stride(M,2)))
    lwork = Int(L) - Int(N)
    lwork ≥ 1 || error("workspace too small for zhseqr: lwork=$lwork")
    work_ptr = pointer(work, N + 1)
    Zdummy = Vector{ComplexF64}(undef, 1)
    ldz = LinearAlgebra.BlasInt(1)
    info = Ref{LinearAlgebra.BlasInt}(0)
    GC.@preserve M W work Zdummy begin
        ccall((:zhseqr_64_, LinearAlgebra.LAPACK.liblapack), Cvoid,
              (Ref{UInt8}, Ref{UInt8}, Ref{LinearAlgebra.BlasInt}, Ref{LinearAlgebra.BlasInt}, Ref{LinearAlgebra.BlasInt}, Ptr{ComplexF64}, Ref{LinearAlgebra.BlasInt},
               Ptr{ComplexF64}, Ptr{ComplexF64}, Ref{LinearAlgebra.BlasInt}, Ptr{ComplexF64}, Ref{LinearAlgebra.BlasInt}, Ref{LinearAlgebra.BlasInt}),
              Ref(UInt8('E')), Ref(UInt8('N')), Ref(n), Ref(LinearAlgebra.BlasInt(ilo)), Ref(LinearAlgebra.BlasInt(ihi)),
              pointer(M), Ref(ldh), pointer(W), pointer(Zdummy), Ref(ldz),
              work_ptr, Ref(LinearAlgebra.BlasInt(lwork)), info)
    end
    info[] == 0 || error("zhseqr failed, info=$(info[])")
    return nothing
end

@inline function _schur_eigvals!(M::MMatrix{N,N,ComplexF32}, ilo::Int, ihi::Int, W::MVector{N,ComplexF32}, work::MVector{L,ComplexF32}) where {N,L}
    n = LinearAlgebra.BlasInt(N)
    ldh = LinearAlgebra.BlasInt(max(1, stride(M,2)))
    lwork = Int(L) - Int(N)
    lwork ≥ 1 || error("workspace too small for chseqr: lwork=$lwork")
    work_ptr = pointer(work, N + 1)
    Zdummy = Vector{ComplexF32}(undef, 1)
    ldz = LinearAlgebra.BlasInt(1)
    info = Ref{LinearAlgebra.BlasInt}(0)
    GC.@preserve M W work Zdummy begin
        ccall((:chseqr_64_, LinearAlgebra.LAPACK.liblapack), Cvoid,
              (Ref{UInt8}, Ref{UInt8}, Ref{LinearAlgebra.BlasInt}, Ref{LinearAlgebra.BlasInt}, Ref{LinearAlgebra.BlasInt}, Ptr{ComplexF32}, Ref{LinearAlgebra.BlasInt},
               Ptr{ComplexF32}, Ptr{ComplexF32}, Ref{LinearAlgebra.BlasInt}, Ptr{ComplexF32}, Ref{LinearAlgebra.BlasInt}, Ref{LinearAlgebra.BlasInt}),
              Ref(UInt8('E')), Ref(UInt8('N')), Ref(n), Ref(LinearAlgebra.BlasInt(ilo)), Ref(LinearAlgebra.BlasInt(ihi)),
              pointer(M), Ref(ldh), pointer(W), pointer(Zdummy), Ref(ldz),
              work_ptr, Ref(LinearAlgebra.BlasInt(lwork)), info)
    end
    info[] == 0 || error("chseqr failed, info=$(info[])")
    return nothing
end