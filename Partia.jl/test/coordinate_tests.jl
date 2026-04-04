using Test
using Partia

import Partia.Tools: _cart2cylin, _cylin2cart,
    _vector_cart2cylin, _vector_cylin2cart,
    _cart2sph, _sph2cart,
    _vector_cart2sph, _vector_sph2cart

@testset "Coordinate transform -- cart <-> cylin round-trip" begin
    x, y = 3.0, 4.0
    s, ϕ = _cart2cylin(x, y)
    @test s ≈ 5.0
    @test 0 <= ϕ < 2π

    x2, y2 = _cylin2cart(s, ϕ)
    @test x2 ≈ x atol = 1e-14
    @test y2 ≈ y atol = 1e-14

    z = 7.0
    s3, ϕ3, z3 = _cart2cylin(x, y, z)
    @test z3 ≈ z
    x3, y3, z3b = _cylin2cart(s3, ϕ3, z3)
    @test x3 ≈ x atol = 1e-14
    @test y3 ≈ y atol = 1e-14
    @test z3b ≈ z atol = 1e-14

    xn, yn = -1.0, -1.0
    sn, ϕn = _cart2cylin(xn, yn)
    @test sn ≈ sqrt(2.0)
    @test π < ϕn < 1.5π
    xn2, yn2 = _cylin2cart(sn, ϕn)
    @test xn2 ≈ xn atol = 1e-14
    @test yn2 ≈ yn atol = 1e-14

    s0, ϕ0 = _cart2cylin(0.0, 0.0)
    @test s0 ≈ 0.0
    @test ϕ0 ≈ 0.0
end

@testset "Coordinate transform -- vector cart <-> cylin round-trip" begin
    ϕ = π / 3
    Ax, Ay = 1.0, 2.0
    As, Aϕ = _vector_cart2cylin(ϕ, Ax, Ay)
    Ax2, Ay2 = _vector_cylin2cart(ϕ, As, Aϕ)
    @test Ax2 ≈ Ax atol = 1e-14
    @test Ay2 ≈ Ay atol = 1e-14

    Az = 5.0
    As3, Aϕ3, Az3 = _vector_cart2cylin(ϕ, Ax, Ay, Az)
    Ax3, Ay3, Az3b = _vector_cylin2cart(ϕ, As3, Aϕ3, Az3)
    @test Az3 ≈ Az
    @test Ax3 ≈ Ax atol = 1e-14
    @test Ay3 ≈ Ay atol = 1e-14
    @test Az3b ≈ Az atol = 1e-14
end

@testset "Coordinate transform -- cart <-> spherical round-trip" begin
    x, y, z = 2.0, -3.0, 6.0
    r, ϕ, θ = _cart2sph(x, y, z)
    @test r ≈ sqrt(x^2 + y^2 + z^2)
    @test 0 <= ϕ < 2π
    @test 0 <= θ <= π

    x2, y2, z2 = _sph2cart(r, ϕ, θ)
    @test x2 ≈ x atol = 1e-14
    @test y2 ≈ y atol = 1e-14
    @test z2 ≈ z atol = 1e-14

    r0, ϕ0, θ0 = _cart2sph(0.0, 0.0, 0.0)
    @test r0 ≈ 0.0
    @test ϕ0 ≈ 0.0
    @test θ0 ≈ 0.0
end

@testset "Coordinate transform -- vector cart <-> spherical round-trip" begin
    x, y, z = 2.0, 3.0, 4.0
    _, ϕ, θ = _cart2sph(x, y, z)

    Ax, Ay, Az = 1.5, -2.0, 0.75
    Ar, Aϕ, Aθ = _vector_cart2sph(ϕ, θ, Ax, Ay, Az)
    Ax2, Ay2, Az2 = _vector_sph2cart(ϕ, θ, Ar, Aϕ, Aθ)
    @test Ax2 ≈ Ax atol = 1e-14
    @test Ay2 ≈ Ay atol = 1e-14
    @test Az2 ≈ Az atol = 1e-14

    Ar3, Aϕ3, Aθ3 = _vector_cart2sph(x, y, z, (Ax, Ay, Az))
    Ax3, Ay3, Az3 = _vector_sph2cart(ϕ, θ, (Ar3, Aϕ3, Aθ3))
    @test Ax3 ≈ Ax atol = 1e-14
    @test Ay3 ≈ Ay atol = 1e-14
    @test Az3 ≈ Az atol = 1e-14
end
