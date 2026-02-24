"""
    tiny_eigvals!(M::MMatrix{N,N,T}) where {N, T<:Complex}

Compute the eigenvalues of a tiny complex square matrix `M` (compile-time size `N`)
using a pure-Julia, fixed-size pipeline that mirrors a simplified LAPACK approach,
specialized for `N ≤ 15`.

Methods are defined for each `N ∈ 1:15` (via metaprogramming) and return the
eigenvalues as a `Tuple{Vararg{T,N}}` to ensure an `isbits` return type and avoid
heap allocation. The input matrix `M` is overwritten in-place by the reduction steps.

Algorithm outline (LAPACK-inspired, simplified, pure Julia):
1. Scaling: scales the matrix to improve numerical robustness.
2. Balancing: permutation/scaling balancing (ZGEBAL('B')-like), returns `ilo, ihi`.
3. Hessenberg reduction: reduces the balanced matrix to upper Hessenberg form
   (ZGEHRD-like), using a small fixed workspace `work`.
4. Schur/QR eigenvalues: QR iterations on the Hessenberg form to obtain eigenvalues
   (ZHSEQR('E','N')-like).
5. Unscaling: rescales the eigenvalues to undo the initial matrix scaling.

# Parameters
- `M::MMatrix{N,N,T}`: Input matrix (modified in-place). `T` must be a complex type.

# Keyword Arguments
None.

# Returns
- `Wout::Tuple{Vararg{T,N}}`: Eigenvalues of `M` as a length-`N` tuple.

# Notes
- Supported sizes: `1 ≤ N ≤ 15`. For other sizes, no method is defined; use
  `LinearAlgebra.eigvals` (LAPACK-backed) instead.
- This is **not** a full LAPACK reimplementation; it follows the same high-level
  stages but is intentionally restricted and tuned for tiny, fixed-size matrices.
- Workspace is fixed-size and stack-allocated via `StaticArrays` to minimize
  allocations in tight loops / batched usage.
"""
function tiny_eigvals! end

for N in 1:15
    @eval begin
        function tiny_eigvals!(M::MMatrix{$N,$N,T}) where {T<:Complex}
            # WORK: length 2N
            work  = zero(MVector{$(2N) ,T})

            # Step 0: scaling (matrix scaling only)
            s = _scale!(M)

            # Step 1: balancing (ZGEBAL('B'))
            ilo, ihi = _balance!(M)

            # Step 2: Hessenberg reduction (ZGEHRD-like)
            # tau stored in work[itau : itau+N-1], scratch in work[iwrk : end]
            _hessenberg_reduce!(M, ilo, ihi, work)

            # Step 3: QR/Schur eigenvalues (ZHSEQR('E','N')-like) & Step 4: undo scaling on eigenvalues
            Wout = _schur_eigvals!(M, ilo, ihi, s, work)
            return Wout
        end
    end
end


