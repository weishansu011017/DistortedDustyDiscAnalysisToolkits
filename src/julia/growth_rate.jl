"""
Growth rate estimation by using Chen & Lin(2020)(doi=10.3847/1538-4357/ab76ca)
    by Wei-Shan Su,
    Augest 5, 2024
"""

"""
    growth_rate(;Κx :: Float64, 
                     Κz :: Float64,
                     St :: Float64,
                     ρg :: Float64,
                     ρd :: Float64, 
                     vx :: Float64,
                     vy :: Float64,
                     ωx :: Float64,
                     ωy :: Float64) :: Float64

Estimation the growth rate of planetesimal with given conditions by using the method in Chen&Lin(2020)((doi=10.3847/1538-4357/ab76ca))

# Parameters
- `Κx`: The dimentionless perturbation wavenumber along x-axis in the shearing box(radial axis)    (Κx = kx * Hg)
- `Κz`: The dimentionless perturbation wavenumber along z-axis in the shearing box(vertical axis)  (Κz = kz * Hg)
- `St`: The Stokes number
- `ρg`: The density of gaseous fluid in the midplane of the disk.
- `ρd`: The density of dusty fluid in the midplane of the disk.
- `vx`: The velocity of gaseous fluid along the x-axis in the shearing box(radial axis)            (vx = vx_true/cs)
- `vy`: The velocity of gaseous fluid along the y-axis in the shearing box(azimuthal axis)         (vy = vy_true/cs)
- `ωx`: The velocity of dusty fluid along the x-axis in the shearing box(radial axis)              (ωx = ωx_true/cs)
- `ωy`: The velocity of dusty fluid along the y-axis in the shearing box(azimuthal axis)           (ωy = ωy_true/cs)

# Return 
- `Float64`: The dimentionless growth rate (s/Ω)(s = Re(σ))
"""
function growth_rate(;Κx :: Float64, 
                     Κz :: Float64,
                     St :: Float64,
                     ρg :: Float64,
                     ρd :: Float64, 
                     vx :: Float64,
                     vy :: Float64,
                     ωx :: Float64,
                     ωy :: Float64) :: Float64

    ε :: Float64 = ρd/ρg
    A :: ComplexF64 = -im * Κx * (ωx)
    B :: ComplexF64 = -im * Κx * (vx)
    Rx :: Float64 = (ε/St) * (ωx - vx)
    Ry :: Float64 = (ε/St) * (ωy - vy)
    invSt :: Float64 = 1/St
    εinvSt :: Float64 = ε * invSt

    M :: Matrix{ComplexF64} = zeros(ComplexF64,8,8)

    M[1,1] = A
    M[1,2] = -im * Κx
    M[1,4] = -im * Κz

    M[2,2] = A - invSt
    M[2,3] = 2
    M[2,6] = invSt
    
    M[3,2] = -0.5
    M[3,3] = A - invSt
    M[3,7] = invSt

    M[4,4] = A - invSt
    M[4,8] = invSt

    M[5,5] = B
    M[5,6] = -im * Κx
    M[5,8] = -im * Κz

    M[6,1] = Rx
    M[6,2] = εinvSt
    M[6,5] = (-im * Κx) - (Rx)
    M[6,6] = B - εinvSt 
    M[6,7] = 2

    M[7,1] = Ry
    M[7,3] = εinvSt
    M[7,5] = -Ry
    M[7,6] = -0.5
    M[7,7] = B - εinvSt

    M[8,4] = εinvSt
    M[8,5] = -im * Κz
    M[8,8] = B - εinvSt
    try
        σs :: Vector{ComplexF64} = eigvals(M)
        ss :: Vector{Float64} = real(σs)
        if isinf(maximum(ss))
            return NaN64
        else
            return maximum(ss)
        end
    catch
        return NaN64
    end
end

"""
    function growth_rate_vectors(;Κx :: AbstractVector, 
                     Κz :: AbstractVector,
                     St :: Float64,
                     ρg :: Float64,
                     ρd :: Float64, 
                     vx :: Float64,
                     vy :: Float64,
                     ωx :: Float64,
                     ωy :: Float64) :: Array

Allowed an vector entrence for Κx and Κz to estimate the growth rate.

# Parameters
- `Κx`: The dimentionless perturbation wavenumber along x-axis in the shearing box(radial axis)    (Κx = kx * Hg)
- `Κz`: The dimentionless perturbation wavenumber along z-axis in the shearing box(vertical axis)  (Κz = kz * Hg)
- `St`: The Stokes number
- `ρg`: The density of gaseous fluid in the midplane of the disk.
- `ρd`: The density of dusty fluid in the midplane of the disk.
- `vx`: The velocity of gaseous fluid along the x-axis in the shearing box(radial axis)            (vx = vx_true/cs)
- `vy`: The velocity of gaseous fluid along the y-axis in the shearing box(azimuthal axis)         (vy = vy_true/cs)
- `ωx`: The velocity of dusty fluid along the x-axis in the shearing box(radial axis)              (ωx = ωx_true/cs)
- `ωy`: The velocity of dusty fluid along the y-axis in the shearing box(azimuthal axis)           (ωy = ωy_true/cs)


# Return 
- `Array`: The dimentionless growth rate (s/Ω)(s = Re(σ))
"""
function growth_rate_vectors(;Κx :: AbstractVector, 
                     Κz :: AbstractVector,
                     St :: Float64,
                     ρg :: Float64,
                     ρd :: Float64, 
                     vx :: Float64,
                     vy :: Float64,
                     ωx :: Float64,
                     ωy :: Float64) :: Array
    s = Array{Float64}(undef,length(Κz),length(Κx))
    @threads for i in eachindex(Κz)
        for j in eachindex(Κx)
            s[i,j] = growth_rate(Κx=Κx[j],Κz=Κz[i],St=St,ρg=ρg,ρd=ρd,vx=vx,vy=vy,ωx=ωx,ωy=ωy)
        end
    end
    return s
end

"""
    growth_rate_input_dict()
Since we fixed the function `growth_rate()` into a fully keyword arguments input-only function, an empty dict for enter the input is presented.

The dictionary will not include "Κx" and "Κz"

# Returns
- `Dict{Symbol, Float64}`: The empty input dictionary with all of the keywords in `growth_rate()`.
"""
function growth_rate_input_dict()
    input :: Dict{Symbol, Float64}= Dict{Symbol, Float64}()
    exclude = [:Κx, :Κz]
    keywords :: Vector = Base.kwarg_decl.(methods(growth_rate))[1]
    for keyword in keywords
        if !(keyword ∈ exclude)
        input[keyword] = 0.0
        end
    end
    return input
end