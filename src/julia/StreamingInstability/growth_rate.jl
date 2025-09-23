"""
Growth rate estimation by using Chen & Lin(2020)(doi=10.3847/1538-4357/ab76ca)
    by Wei-Shan Su,
    Augest 5, 2024
"""

struct _QR8buffer{T <: Number, V <: AbstractVector{T}, M <: AbstractMatrix{T}}
    A :: M
    reflector :: V
    Q :: M
    B :: M
end

const _QR8buffer_pool = Vector{_QR8buffer{ComplexF64, MVector{8, ComplexF64}, MMatrix{8,8,ComplexF64,64}}}()

function init_QR8buffer_bufferl!()
    resize!(_QR8buffer_pool, nthreads())
    for tid in 1:nthreads()
        _QR8buffer_pool[tid] = _QR8buffer(
            MMatrix{8, 8, ComplexF64, 64}(undef),      
            MVector{8, ComplexF64}(undef),
            MMatrix{8, 8, ComplexF64, 64}(undef),
            MMatrix{8, 8, ComplexF64, 64}(undef),      
        )
    end
end


@inline function _realλ_max(M)
    buf  = _QR8buffer_pool[Threads.threadid()]

    buf.reflector .= LinearAlgebra.LAPACK.geev!('N','N',M)[1]

    λmax = -Inf
    @inbounds @simd for k = 1:8
        r = real(buf.reflector[k])
        λmax = r > λmax ? r : λmax
    end
    return λmax
end

@inline function _growthrateSI_core(Κx :: Float64,
                                   Κz :: Float64,
                                   invSt :: Float64,
                                   εinvSt :: Float64,
                                   vx :: Float64,
                                   vy :: Float64,
                                   ωx :: Float64,
                                   ωy :: Float64) :: Float64

    buf = _QR8buffer_pool[threadid()]
    M     = buf.A
    
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
    if any(x -> isnan(x) || isinf(x), M)
        return NaN
    else
        return _realλ_max(M)
    end
    
    # return maximum(real.(eigvals(M)))
end

"""
    growthrateSI(Κx :: Float64, Κz :: Float64;
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
    growthrateSI(Κxs :: AbstractVector{Float64}, Κzs :: AbstractVector{Float64};
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

"""
    growthrateSI!(ΚxΚzSpace :: AbstractMatrix{Float64}, Κxs :: AbstractVector{Float64}, Κzs :: AbstractVector{Float64};
                         St :: Float64,
                         ρg :: Float64,
                         ρd :: Float64, 
                         vx :: Float64,
                         vy :: Float64,
                         ωx :: Float64,
                         ωy :: Float64) :: Array{Float64}

In-place version of `growthrateSI`. Estimate the dimensionless dust growth rate (s/Ω) under the framework of Streaming Instability (Yooding & Goodman 2005).
Using the method described in Chen & Lin (2020, ApJ, 892, 114), doi:10.3847/1538-4357/ab76ca.

Allowed an vector entrence for Κx and Κz to estimate the growth rate.

# Parameters
- `ΚxΚzSpace::AbstractMatrix{Float64}`: Preallocated 2D array to store output growth rates. Must be of shape `(length(Κxs), length(Κzs))`.
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
"""
function growthrateSI!(ΚxΚzSpace :: AbstractMatrix{Float64}, Κxs :: AbstractVector{Float64}, Κzs :: AbstractVector{Float64};
    St :: Float64,
    ρg :: Float64,
    ρd :: Float64, 
    vx :: Float64,
    vy :: Float64,
    ωx :: Float64,
    ωy :: Float64)
    ε = ρd/ρg
    invSt = 1 / St
    εinvSt = ε * invSt

    @inbounds for j in eachindex(Κzs), i in eachindex(Κxs)
        ΚxΚzSpace[i,j] = _growthrateSI_core(Κxs[i], Κzs[j], invSt, εinvSt, vx, vy, ωx, ωy)
    end
end

