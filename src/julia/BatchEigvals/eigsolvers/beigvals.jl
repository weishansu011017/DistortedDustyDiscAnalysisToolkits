

@generated function beigvals!(M::MMatrix{N,N,T}) where {N, T<:Complex}
    if N > 15
        return quote
            error("DimensionError: N = $N > 15, this implementation is for small N only. Use the package `LinearAlgebra` instead!")
        end
    elseif N < 1
        return quote
            error("DimensionError: N = $N < 1, Empty matrix is not allowed")
        end
    else
        worksize = 33 * N
        RT = _realtype(T)               
        quote

            W = zero(MVector{$N,T})

            # WORK: length 33N (ZGEEV worst-case for N<=15)
            work  = zero(MVector{$worksize,T})

            # RWORK: keep N for balancing factors (ZGEBAL)
            rwork = zero(MVector{$N,$RT})

            # Step 0: scaling (matrix scaling only)
            scaled, s = _scale!(M)

            # Step 1: balancing (ZGEBAL('B'))
            ilo, ihi = _balance!(M, rwork)

            # Step 2: Hessenberg reduction (ZGEHRD-like)
            # tau stored in work[itau : itau+N-1], scratch in work[iwrk : end]
            _hessenberg_reduce!(M, ilo, ihi, work)

            # Step 3: QR/Schur eigenvalues (ZHSEQR('E','N')-like)
            _schur_eigvals!(M, ilo, ihi, W, work)

            # Step 4: undo scaling on eigenvalues
            scaled && _unscale!(W, s)

            return W
        end
    end
end



