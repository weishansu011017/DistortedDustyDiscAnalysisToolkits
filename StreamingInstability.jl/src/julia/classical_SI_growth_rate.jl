"""
Growth rate estimation by using Chen & Lin(2020)(doi=10.3847/1538-4357/ab76ca)
    by Wei-Shan Su,
    Augest 5, 2024
"""

"""
# Fields
| Name    | Description                                                                 |
|---------|-----------------------------------------------------------------------------|
| `St`    | Stokes number of dust particles.                                            |
| `ρg`    | Midplane gas density.                                                       |
| `ρd`    | Midplane dust density.                                                      |
| `vxlcs` | Dimensionless gas velocity along radial (x) axis IN SOUND SPEED (vxlcs = vx_true / c_s).    |
| `vylcs` | Dimensionless gas velocity along azimuthal (y) axis IN SOUND SPEED (vylcs = vy_true / c_s). |
| `ωxlcs` | Dimensionless dust velocity along radial (x) axis IN SOUND SPEED (ωxlcs = ωx_true / c_s).   |
| `ωylcs` | Dimensionless dust velocity along azimuthal (y) axis IN SOUND SPEED (ωylcs = ωy_true / c_s).|
"""
struct ClassicalSIGrowthRateInput{T <: AbstractFloat}
    St      ::  T
    ρg      ::  T
    ρd      ::  T
    vxlcs   ::  T
    vylcs   ::  T
    ωxlcs   ::  T
    ωylcs   ::  T

    # Internal parameters
    _invSt  :: T        # inverse of St
    _εinvSt :: T        # (Midplane) Dust-to-Gas Ratio times invSt
end

function ClassicalSIGrowthRateInput(St      ::  T,
                                    ρg      ::  T,
                                    ρd      ::  T,
                                    vxlcs   ::  T,
                                    vylcs   ::  T,
                                    ωxlcs   ::  T,
                                    ωylcs   ::  T) where {T <: AbstractFloat}
    ε = ρd/ρg
    invSt = 1 / St
    εinvSt = ε * invSt
    return ClassicalSIGrowthRateInput(St, ρg, ρd, vxlcs, vylcs, ωxlcs, ωylcs, invSt, εinvSt)
end

@inline function _realλ_max(M :: MMatrix{N, N, ComplexF64}) where {N}
    eigenvalues = tiny_eigvals!(M)

    λmax = -Inf
    @inbounds @simd for k = SOneTo(N)
        r = real(eigenvalues[k])
        λmax = ifelse(r > λmax, r, λmax)
    end
    return λmax
end

"""
    (CSIGRInput :: ClassicalSIGrowthRateInput{Float64})(Κx :: Float64, Κz :: Float64) :: Float64

Estimate the dimensionless dust growth rate (s/Ω) under the framework of Streaming Instability (Yooding & Goodman 2005).
Using the method described in Chen & Lin (2020, ApJ, 892, 114), doi:10.3847/1538-4357/ab76ca.

# Parameters
- `CSIGRInput :: ClassicalSIGrowthRateInput{Float64}`: The other input parameters for estimating growth rate.
- `Κx         :: Float64`: Dimensionless radial wavenumber in the shearing box   (Κx = kx × Hg).
- `Κz         :: Float64`: Dimensionless vertical wavenumber in the shearing box (Κz = kz × Hg).

# Return 
- `Float64`: Array of dimentionless growth rate (s/Ω)(s = Re(σ)) with size = (length(Κxs), length(Κzs))
"""
@inline function (CSIGRInput :: ClassicalSIGrowthRateInput{Float64})(Κx :: Float64, Κz :: Float64) :: Float64
    vx = CSIGRInput.vxlcs
    vy = CSIGRInput.vylcs
    ωx = CSIGRInput.ωxlcs
    ωy = CSIGRInput.ωylcs
    invSt = CSIGRInput._invSt
    εinvSt = CSIGRInput._εinvSt

    Rx, Ry = εinvSt*(ωx-vx), εinvSt*(ωy-vy)
    A = -im*Κx*ωx
    B = -im*Κx*vx

    # Check all the parameters is finite
    if any(x -> !isfinite(x), (Κx, Κz, invSt, εinvSt, Rx, Ry, A, B))
        return NaN
    else
        M = zero(MMatrix{8, 8, ComplexF64})
        @inbounds begin
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
        return _realλ_max(M)
    end
end

"""
    (CSIGRInput :: ClassicalSIGrowthRateInput{Float64})(SIgrowth :: M, Κxs :: V, Κzs :: V) where {V <: AbstractVector{Float64}, M <: AbstractMatrix{Float64}}

Estimate the dimensionless dust growth rate (s/Ω) under the framework of Streaming Instability (Yooding & Goodman 2005).
Using the method described in Chen & Lin (2020, ApJ, 892, 114), doi:10.3847/1538-4357/ab76ca.

Allowed an vector entrence for Κx and Κz to estimate the growth rate.

# Parameters
- `CSIGRInput :: ClassicalSIGrowthRateInput{Float64}`: The other input parameters for estimating growth rate.
- `ΚxΚzSpace :: AbstractMatrix{Float64}`: Preallocated 2D array to store output growth rates. Must be of shape `(length(Κxs), length(Κzs))`.
- `Κxs :: AbstractVector{Float64}`: Array of dimensionless radial wavenumbers in the shearing box   (Κx = kx × Hg).
- `Κzs :: AbstractVector{Float64}`: Array of dimensionless vertical wavenumbers in the shearing box (Κz = kz × Hg).
"""
@inline function (CSIGRInput :: ClassicalSIGrowthRateInput{Float64})(SIgrowth :: MF, Κxs :: V, Κzs :: V) where {V <: AbstractVector{Float64}, MF <: AbstractMatrix{Float64}}
    vx = CSIGRInput.vxlcs
    vy = CSIGRInput.vylcs
    ωx = CSIGRInput.ωxlcs
    ωy = CSIGRInput.ωylcs
    invSt = CSIGRInput._invSt
    εinvSt = CSIGRInput._εinvSt

    Rx, Ry = εinvSt*(ωx-vx), εinvSt*(ωy-vy)
    # Check most of the parameters is finite
    if any(x -> !isfinite(x), (invSt, εinvSt, Rx, Ry))
        fill!(SIgrowth, NaN)
    else
        M = zero(MMatrix{8, 8, ComplexF64})
        @inbounds for j in eachindex(Κzs), i in eachindex(Κxs)
            Κx = Κxs[i]
            Κz = Κzs[j]
            A = -im*Κx*ωx
            B = -im*Κx*vx
            if any(x -> !isfinite(x), (Κx, Κz, A, B))
                SIgrowth[i,j] = NaN
            else
                @inbounds begin
                    fill!(M, zero(ComplexF64))
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
                SIgrowth[i,j] = _realλ_max(M)
            end
        end
    end
    return nothing
end

"""
    (CSIGRInput :: ClassicalSIGrowthRateInput{Float64})(Κxs :: V, Κzs :: V) where {V <: AbstractVector{Float64}}

Estimate the dimensionless dust growth rate (s/Ω) under the framework of Streaming Instability (Yooding & Goodman 2005).
Using the method described in Chen & Lin (2020, ApJ, 892, 114), doi:10.3847/1538-4357/ab76ca.

Allowed an vector entrence for Κx and Κz to estimate the growth rate.

# Parameters
- `CSIGRInput :: ClassicalSIGrowthRateInput{Float64}`: The other input parameters for estimating growth rate.
- `Κxs :: AbstractVector{Float64}`: Array of dimensionless radial wavenumbers in the shearing box   (Κx = kx × Hg).
- `Κzs :: AbstractVector{Float64}`: Array of dimensionless vertical wavenumbers in the shearing box (Κz = kz × Hg).
"""
@inline function (CSIGRInput :: ClassicalSIGrowthRateInput{Float64})(Κxs :: V, Κzs :: V) where {V <: AbstractVector{Float64}}
    SIgrowth = zeros(Float64, length(Κxs), length(Κzs))
    CSIGRInput(SIgrowth, Κxs, Κzs)
    return SIgrowth
end