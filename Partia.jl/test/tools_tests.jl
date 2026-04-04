using Test
using Partia

@testset "SoundSpeed -- Adiabatic" begin
    γ = 5.0 / 3.0
    u = 2.0
    cs = SoundSpeed(Adiabatic, u, γ)
    @test cs ≈ sqrt(γ * (γ - 1) * u)

    cs_mixed = SoundSpeed(Adiabatic, 2.0f0, γ)
    @test cs_mixed isa Float64

    @test isnan(SoundSpeed(Adiabatic, -1.0, γ))
    @test isnan(SoundSpeed(Adiabatic, 2.0, 0.5))
end

@testset "SoundSpeed -- Isothermal" begin
    cs0 = 0.3
    @test SoundSpeed(Isothermal, cs0) ≈ cs0
end

@testset "SoundSpeed -- LocallyIsothermal" begin
    cs0 = 1.0
    r = 2.0
    q = 0.25
    cs = SoundSpeed(LocallyIsothermal, r, cs0, q)
    @test cs ≈ cs0 * r^(-q)

    @test isnan(SoundSpeed(LocallyIsothermal, -1.0, cs0, q))
    @test isnan(SoundSpeed(LocallyIsothermal, 0.0, cs0, q))
end

@testset "Pressure -- Adiabatic" begin
    γ = 5.0 / 3.0
    ρ = 1.5
    u = 2.0
    P = Pressure(Adiabatic, ρ, u, γ)
    @test P ≈ (γ - 1) * ρ * u

    @test isnan(Pressure(Adiabatic, -1.0, u, γ))
    @test isnan(Pressure(Adiabatic, ρ, -1.0, γ))
    @test isnan(Pressure(Adiabatic, ρ, u, 0.5))
end

@testset "Pressure -- Isothermal" begin
    ρ = 1.0
    cs = 0.3
    @test Pressure(Isothermal, ρ, cs) ≈ ρ * cs^2

    @test isnan(Pressure(Isothermal, -1.0, cs))
end

@testset "Pressure -- LocallyIsothermal" begin
    ρ = 1.0
    r = 2.0
    cs0 = 1.0
    q = 0.25
    P = Pressure(LocallyIsothermal, ρ, r, cs0, q)
    @test P ≈ ρ * (cs0 * r^(-q))^2

    @test isnan(Pressure(LocallyIsothermal, -1.0, r, cs0, q))
    @test isnan(Pressure(LocallyIsothermal, ρ, 0.0, cs0, q))
end

@testset "meshgrid -- 2D and 3D" begin
    xs = [1.0, 2.0, 3.0]
    ys = [10.0, 20.0]

    X, Y = meshgrid(xs, ys)
    @test size(X) == (3, 2)
    @test size(Y) == (3, 2)
    @test X[:, 1] == xs
    @test X[:, 2] == xs
    @test Y[1, :] == ys
    @test Y[3, :] == ys

    zs = [100.0]
    X3, Y3, Z3 = meshgrid(xs, ys, zs)
    @test size(X3) == (3, 2, 1)
    @test size(Y3) == (3, 2, 1)
    @test size(Z3) == (3, 2, 1)
end

@testset "NaN-safe statistics" begin
    A = [1.0, NaN, 3.0, NaN, 5.0]

    @test nanmean(A) ≈ 3.0
    @test nanmaximum(A) ≈ 5.0
    @test nanminimum(A) ≈ 1.0
    @test isnan(nanmean([NaN, NaN]))

    B = [2.0, 4.0, 6.0]
    @test nanmean(B) ≈ 4.0
    @test nanstd(B) ≈ 2.0
end

@testset "Euclidean distance" begin
    x = [0.0, 3.0]
    y = [0.0, 4.0]
    z = [0.0, 0.0]

    d3 = Euclidean_distance(x, y, z, (0.0, 0.0, 0.0))
    @test d3[1] ≈ 0.0
    @test d3[2] ≈ 5.0

    d2 = Euclidean_distance(x, y, (0.0, 0.0))
    @test d2[1] ≈ 0.0
    @test d2[2] ≈ 5.0
end

@testset "invert_order -- inverse permutation identity" begin
    order = [3, 1, 4, 2]
    inv_order = invert_order(order)
    v = [10, 20, 30, 40]
    @test v[order][inv_order] == v
end

@testset "maxabs -- complex array" begin
    A = ComplexF64[1 + 2im, 3 + 4im, -5 + 0im]
    @test maxabs(A) ≈ 5.0
end
