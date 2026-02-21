@inline function _hessenberg_reduce!(M::MMatrix{N,N,ComplexF64}, ilo::Int, ihi::Int, work::MVector{L,ComplexF64}) where {N,L}
    n = LinearAlgebra.BlasInt(N)
    lda = LinearAlgebra.BlasInt(max(1, stride(M,2)))
    lwork = Int(L) - Int(N)
    lwork ≥ 1 || error("workspace too small for zgehrd: lwork=$lwork")
    tau_ptr = pointer(work)                # first N entries
    work_ptr = pointer(work, N + 1)        # remaining workspace
    info = Ref{LinearAlgebra.BlasInt}(0)
    GC.@preserve M work begin
          ccall((:zgehrd_64_, LinearAlgebra.LAPACK.liblapack), Cvoid,
              (Ref{LinearAlgebra.BlasInt}, Ref{LinearAlgebra.BlasInt}, Ref{LinearAlgebra.BlasInt}, Ptr{ComplexF64}, Ref{LinearAlgebra.BlasInt}, Ptr{ComplexF64}, Ptr{ComplexF64}, Ref{LinearAlgebra.BlasInt}, Ref{LinearAlgebra.BlasInt}),
              Ref(n), Ref(LinearAlgebra.BlasInt(ilo)), Ref(LinearAlgebra.BlasInt(ihi)), pointer(M), Ref(lda),
              tau_ptr, work_ptr, Ref(LinearAlgebra.BlasInt(lwork)), info)
    end
    info[] == 0 || error("zgehrd failed, info=$(info[])")
    return nothing
end

@inline function _hessenberg_reduce!(M::MMatrix{N,N,ComplexF32}, ilo::Int, ihi::Int, work::MVector{L,ComplexF32}) where {N,L}
    n = LinearAlgebra.BlasInt(N)
    lda = LinearAlgebra.BlasInt(max(1, stride(M,2)))
    lwork = Int(L) - Int(N)
    lwork ≥ 1 || error("workspace too small for cgehrd: lwork=$lwork")
    tau_ptr = pointer(work)
    work_ptr = pointer(work, N + 1)
    info = Ref{LinearAlgebra.BlasInt}(0)
    GC.@preserve M work begin
          ccall((:cgehrd_64_, LinearAlgebra.LAPACK.liblapack), Cvoid,
              (Ref{LinearAlgebra.BlasInt}, Ref{LinearAlgebra.BlasInt}, Ref{LinearAlgebra.BlasInt}, Ptr{ComplexF32}, Ref{LinearAlgebra.BlasInt}, Ptr{ComplexF32}, Ptr{ComplexF32}, Ref{LinearAlgebra.BlasInt}, Ref{LinearAlgebra.BlasInt}),
              Ref(n), Ref(LinearAlgebra.BlasInt(ilo)), Ref(LinearAlgebra.BlasInt(ihi)), pointer(M), Ref(lda),
              tau_ptr, work_ptr, Ref(LinearAlgebra.BlasInt(lwork)), info)
    end
    info[] == 0 || error("cgehrd failed, info=$(info[])")
    return nothing
end