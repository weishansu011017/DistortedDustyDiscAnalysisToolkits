"""
Growth rate estimation by using Chen & Lin(2020)(doi=10.3847/1538-4357/ab76ca)
    by Wei-Shan Su,
    Augest 5, 2024
"""

struct SIgrowthrate_requirements
    Κxs :: Vector{Float64}
    Κzs :: Vector{Float64}
    St :: Float64
    ρg :: Float64
    ρd::Float64
    vx::Float64
    vy::Float64
    ωx::Float64
    ωy::Float64
end

struct GrowthBuf
    M :: Matrix{ComplexF64}
    MW  :: MMatrix{8,8,ComplexF64,64}     
    LU :: MMatrix{8,8,ComplexF64,64}      
    v  :: MVector{8,ComplexF64}           
    w  :: MVector{8,ComplexF64}          
    invU :: MVector{8,ComplexF64}        
end

const _growth_pool = Vector{GrowthBuf}()

function init_growth_buffers!()
    resize!(_growth_pool, nthreads())
    for tid in 1:nthreads()
        _growth_pool[tid] = GrowthBuf(
            zeros(ComplexF64, 8, 8),
            zeros(MMatrix{8,8,ComplexF64,64}),
            zeros(MMatrix{8,8,ComplexF64,64}),
            MVector{8,ComplexF64}(1, 0, 0, 0, 0, 0, 0, 0),
            MVector{8,ComplexF64}(undef),
            MVector{8,ComplexF64}(undef),
        )
    end
end

@inline _nrm2_8(v) = begin
    s = 0.0
    @inbounds @simd for i = 1:8
        s += abs2(v[i])
    end
    sqrt(s)
end

@inline _dot8(a, b) = begin
    s = 0.0 + 0.0im
    @inbounds @simd for i=1:8
        s += a[i] * conj(b[i])
    end
    s
end

@inline function _lu8_nopiv_inv!(A, invU)
    @inbounds @fastmath for k = 1:7
        inv_d      = inv(A[k,k])
        invU[k]    = inv_d
        for i = k+1:8
            lik = A[i,k] *= inv_d
            @simd for j = k+1:8
                A[i,j] -= lik * A[k,j]
            end
        end
    end
    invU[8] = inv(A[8,8])
end

@inline function _fwd8!(LU, x)
    @inbounds @fastmath begin
        x2 = x[2] -= LU[2,1]*x[1]
        x3 = x[3] -= LU[3,1]*x[1] + LU[3,2]*x2
        x4 = x[4] -= LU[4,1]*x[1] + LU[4,2]*x2 + LU[4,3]*x3
        x5 = x[5] -= LU[5,1]*x[1] + LU[5,2]*x2 + LU[5,3]*x3 + LU[5,4]*x4
        x6 = x[6] -= LU[6,1]*x[1] + LU[6,2]*x2 + LU[6,3]*x3 + LU[6,4]*x4 + LU[6,5]*x5
        x7 = x[7] -= LU[7,1]*x[1] + LU[7,2]*x2 + LU[7,3]*x3 + LU[7,4]*x4 + LU[7,5]*x5 + LU[7,6]*x6
        x8 = x[8] -= LU[8,1]*x[1] + LU[8,2]*x2 + LU[8,3]*x3 + LU[8,4]*x4 + LU[8,5]*x5 + LU[8,6]*x6 + LU[8,7]*x7
    end
end

@inline function _bwd8!(LU, x, invU)
    @inbounds @fastmath begin
        x8 = x[8] *= invU[8]                                
        x7 = x[7] = (x[7] - LU[7,8]*x8) * invU[7]
        x6 = x[6] = (x[6] - LU[6,8]*x8 - LU[6,7]*x7) * invU[6]
        x5 = x[5] = (x[5] - LU[5,8]*x8 - LU[5,7]*x7 - LU[5,6]*x6) * invU[5]
        x4 = x[4] = (x[4] - LU[4,8]*x8 - LU[4,7]*x7 - LU[4,6]*x6 - LU[4,5]*x5) * invU[4]
        x3 = x[3] = (x[3] - LU[3,8]*x8 - LU[3,7]*x7 - LU[3,6]*x6 - LU[3,5]*x5 - LU[3,4]*x4) * invU[3]
        x2 = x[2] = (x[2] - LU[2,8]*x8 - LU[2,7]*x7 - LU[2,6]*x6 - LU[2,5]*x5 - LU[2,4]*x4 - LU[2,3]*x3) * invU[2]
        x1 = x[1] = (x[1] - LU[1,8]*x8 - LU[1,7]*x7 - LU[1,6]*x6 - LU[1,5]*x5 - LU[1,4]*x4 - LU[1,3]*x3 - LU[1,2]*x2) * invU[1]
    end
end

@inline _trisolve8_nopiv!(LU, x, invU) = (_fwd8!(LU,x); _bwd8!(LU,x, invU))

@inline function _realλ_max(;maxiter=16, atol::Float64 = 5e-7, rtol::Float64 = 1e-8) :: Float64
    buf = _growth_pool[threadid()]
    MW = buf.MW
    v  = buf.v
    w  = buf.w
    LU = buf.LU
    invU = buf.invU

    fill!(v, 0.0)
    v[1] = 1.0 + 0.0im

    σ = 0.1
    μ  = 0 + 0im
    μ_old = Inf + 0im  
    λ = 0.0
    inner_max = 3
    outer_max = maxiter ÷ inner_max       

    for _o = 1:outer_max
        # --- (A − σI) LU ---
        copyto!(LU, MW)
        @inbounds for i=1:8 LU[i,i] -= σ end
        _lu8_nopiv_inv!(LU, invU)

        for _i = 1:inner_max
            w .= v
            _trisolve8_nopiv!(LU, w, invU)

            μ = _dot8(v, w)               
            abs(μ) ≤ 1e-300 && return -Inf
            λ = σ + inv(μ)

            # residual
            rsq = 0.0
            @inbounds @fastmath @simd for k=1:8
                tmp  = w[k] - μ*v[k]
                rsq += abs2(tmp)
            end
            r = sqrt(rsq) / abs(μ)
            if (r < atol) && (abs(μ-μ_old)/abs(μ) < rtol)
                return real(λ)
            end

            # normalize v
            invn = 1 / _nrm2_8(w)
            @inbounds @fastmath @simd for k=1:8
                v[k] = w[k] * invn
            end
            μ_old = μ
        end
        σ = real(λ)                         
    end
    return σ
end

@inline function _growthrateSI_core(Κx :: Float64,
                                   Κz :: Float64,
                                   invSt :: Float64,
                                   εinvSt :: Float64,
                                   vx :: Float64,
                                   vy :: Float64,
                                   ωx :: Float64,
                                   ωy :: Float64) :: Float64

    buf = _growth_pool[threadid()]
    M     = buf.M
    MW = buf.MW            
    
    Rx, Ry = εinvSt*(ωx-vx), εinvSt*(ωy-vy)
    A = -im*Κx*ωx
    B = -im*Κx*vx
    @inbounds begin
        fill!(M, 0.0 + 0.0im) 

        M[1,1] = A
        M[6,1] = Rx
        M[7,1] = Ry

        M[1,2] = -im * Κx
        M[2,2] = A - invSt
        M[3,2] = -0.5
        M[6,2] = εinvSt

        M[2,3] = 2
        M[3,3] = A - invSt
        M[7,3] = εinvSt

        M[1,4] = -im * Κz
        M[4,4] = A - invSt
        M[8,4] = εinvSt

        M[5,5] = B
        M[6,5] = (-im * Κx) - (Rx)
        M[7,5] = -Ry
        M[8,5] = -im * Κz

        M[2,6] = invSt
        M[5,6] = -im * Κx
        M[6,6] = B - εinvSt
        M[7,6] = -0.5

        M[3,7] = invSt
        M[6,7] = 2
        M[7,7] = B - εinvSt

        M[4,8] = invSt
        M[5,8] = -im * Κz
        M[8,8] = B - εinvSt
    end
    @inbounds MW .= M
    return _realλ_max()
end

"""
    function growthrateSI(Κx :: Float64, Κz :: Float64;
                          St :: Float64,
                          ρg :: Float64,
                          ρd :: Float64, 
                          vx :: Float64,
                          vy :: Float64,
                          ωx :: Float64,
                          ωy :: Float64) :: Float64

Estimate the dimensionless dust growth rate (s/Ω) under the framework of Streaming Instability (Yooding & Goodman 2005).
Using the method described in Chen & Lin (2020, ApJ, 892, 114), doi:10.3847/1538-4357/ab76ca.

# Parameters
- `Κx::Float64`: Dimensionless radial wavenumber in the shearing box   (Κx = kx × Hg).
- `Κz::Float64`: Dimensionless vertical wavenumber in the shearing box (Κz = kz × Hg).

# Keyword Arguments
| Name   | Description                                                                 |
|--------|-----------------------------------------------------------------------------|
| `St`   | Stokes number of dust particles.                                            |
| `ρg`   | Midplane gas density.                                                       |
| `ρd`   | Midplane dust density.                                                      |
| `vx`   | Dimensionless gas velocity along radial (x) axis IN SOUND SPEED (vx = vx_true / c_s).    |
| `vy`   | Dimensionless gas velocity along azimuthal (y) axis IN SOUND SPEED (vy = vy_true / c_s). |
| `ωx`   | Dimensionless dust velocity along radial (x) axis IN SOUND SPEED (ωx = ωx_true / c_s).   |
| `ωy`   | Dimensionless dust velocity along azimuthal (y) axis IN SOUND SPEED (ωy = ωy_true / c_s).|


# Return 
- `Float64`: Array of dimentionless growth rate (s/Ω)(s = Re(σ)) with size = (length(Κxs), length(Κzs))
"""
function growthrateSI(Κx :: Float64, Κz :: Float64;
                      St :: Float64,
                      ρg :: Float64,
                      ρd :: Float64, 
                      vx :: Float64,
                      vy :: Float64,
                      ωx :: Float64,
                      ωy :: Float64) :: Float64
    ε = ρd/ρg
    invSt = 1 / St
    εinvSt = ε * invSt
    return _growthrateSI_core(Κx, Κz, invSt, εinvSt, vx, vy, ωx, ωy)
end

"""
    function growthrateSI(Κxs :: AbstractVector{Float64}, Κzs :: AbstractVector{Float64};
                          St :: Float64,
                          ρg :: Float64,
                          ρd :: Float64, 
                          vx :: Float64,
                          vy :: Float64,
                          ωx :: Float64,
                          ωy :: Float64) :: Array{Float64}

Estimate the dimensionless dust growth rate (s/Ω) under the framework of Streaming Instability (Yooding & Goodman 2005).
Using the method described in Chen & Lin (2020, ApJ, 892, 114), doi:10.3847/1538-4357/ab76ca.

Allowed an vector entrence for Κx and Κz to estimate the growth rate.

# Parameters
- `Κxs::AbstractVector{Float64}`: Array of dimensionless radial wavenumbers in the shearing box   (Κx = kx × Hg).
- `Κzs::AbstractVector{Float64}`: Array of dimensionless vertical wavenumbers in the shearing box (Κz = kz × Hg).

# Keyword Arguments
| Name   | Description                                                                 |
|--------|-----------------------------------------------------------------------------|
| `St`   | Stokes number of dust particles.                                            |
| `ρg`   | Midplane gas density.                                                       |
| `ρd`   | Midplane dust density.                                                      |
| `vx`   | Dimensionless gas velocity along radial (x) axis IN SOUND SPEED (vx = vx_true / c_s).    |
| `vy`   | Dimensionless gas velocity along azimuthal (y) axis IN SOUND SPEED (vy = vy_true / c_s). |
| `ωx`   | Dimensionless dust velocity along radial (x) axis IN SOUND SPEED (ωx = ωx_true / c_s).   |
| `ωy`   | Dimensionless dust velocity along azimuthal (y) axis IN SOUND SPEED (ωy = ωy_true / c_s).|


# Return 
- `Array`: Array of dimentionless growth rate (s/Ω)(s = Re(σ)) with size = (length(Κxs), length(Κzs))
"""
function growthrateSI(Κxs :: AbstractVector{Float64}, Κzs :: AbstractVector{Float64};
                      St :: Float64,
                      ρg :: Float64,
                      ρd :: Float64, 
                      vx :: Float64,
                      vy :: Float64,
                      ωx :: Float64,
                      ωy :: Float64) :: Array{Float64}
    SIgrowth = zeros(Float64, length(Κxs), length(Κzs))
    ε = ρd/ρg
    invSt = 1 / St
    εinvSt = ε * invSt

    @inbounds for j in eachindex(Κzs), i in eachindex(Κxs)
        SIgrowth[i,j] = _growthrateSI_core(Κxs[i], Κzs[j], invSt, εinvSt, vx, vy, ωx, ωy)
    end
    return SIgrowth
end

