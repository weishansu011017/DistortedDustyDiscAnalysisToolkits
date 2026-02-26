# ──────────────────────────────────────────────────────────────────────────── #
#  Test: Tools — EOS, Coordinate Transforms, and Array Utilities
# ──────────────────────────────────────────────────────────────────────────── #
#
#  What this file tests
#  ─────────────────────
#  Validates utility functions exported by the `Tools` submodule:
#
#  1. Equation of State (EOS)
#     • SoundSpeed  — Adiabatic, Isothermal, LocallyIsothermal.
#     • Pressure    — Adiabatic, Isothermal, LocallyIsothermal.
#     • NaN guards  — unphysical inputs (ρ < 0, u < 0, γ < 1, r ≤ 0).
#
#  2. Coordinate transformations
#     • _cart2cylin / _cylin2cart — round-trip identity.
#     • _vector_cart2cylin / _vector_cylin2cart — round-trip identity.
#     • Special cases: origin, axis-aligned points.
#
#  3. Array utilities
#     • meshgrid — shape and value checks for 2D / 3D.
#     • nanmean / nanstd / nanmaximum / nanminimum — NaN-skipping.
#     • Euclidean_distance — 2D and 3D.
#     • invert_order — inverse permutation identity.
#     • maxabs — complex arrays.
#
#  Reference formulas
#  ──────────────────
#  • Adiabatic:         cₛ = √(γ(γ−1)u),  P = (γ−1)ρu
#  • Isothermal:        cₛ = cₛ,           P = ρ cₛ²
#  • LocallyIsothermal: cₛ = cₛ₀ r⁻ᵠ,     P = ρ (cₛ₀ r⁻ᵠ)²
#
# ──────────────────────────────────────────────────────────────────────────── #

using Test
using PhantomRevealer
import PhantomRevealer.Tools: _cart2cylin, _cylin2cart,
    _vector_cart2cylin, _vector_cylin2cart

# ============================== Test body =================================== #

# ── 1a. SoundSpeed ───────────────────────────────────────────────────── #

@testset "SoundSpeed — Adiabatic" begin
    γ = 5.0 / 3.0
    u = 2.0
    cs = SoundSpeed(Adiabatic, u, γ)
    @test cs ≈ sqrt(γ * (γ - 1) * u)

    # Type promotion: Float32 + Float64
    cs_mixed = SoundSpeed(Adiabatic, 2.0f0, γ)
    @test cs_mixed isa Float64

    # NaN guards
    @test isnan(SoundSpeed(Adiabatic, -1.0, γ))    # u < 0
    @test isnan(SoundSpeed(Adiabatic, 2.0, 0.5))    # γ < 1
end

@testset "SoundSpeed — Isothermal" begin
    cs0 = 0.3
    @test SoundSpeed(Isothermal, cs0) ≈ cs0
end

@testset "SoundSpeed — LocallyIsothermal" begin
    cs0 = 1.0;  r = 2.0;  q = 0.25
    cs = SoundSpeed(LocallyIsothermal, r, cs0, q)
    @test cs ≈ cs0 * r^(-q)

    # NaN guards
    @test isnan(SoundSpeed(LocallyIsothermal, -1.0, cs0, q))  # r ≤ 0
    @test isnan(SoundSpeed(LocallyIsothermal, 0.0, cs0, q))   # r = 0
end

# ── 1b. Pressure ─────────────────────────────────────────────────────── #

@testset "Pressure — Adiabatic" begin
    γ = 5.0 / 3.0;  ρ = 1.5;  u = 2.0
    P = Pressure(Adiabatic, ρ, u, γ)
    @test P ≈ (γ - 1) * ρ * u

    # NaN guards
    @test isnan(Pressure(Adiabatic, -1.0, u, γ))
    @test isnan(Pressure(Adiabatic, ρ, -1.0, γ))
    @test isnan(Pressure(Adiabatic, ρ, u, 0.5))
end

@testset "Pressure — Isothermal" begin
    ρ = 1.0;  cs = 0.3
    @test Pressure(Isothermal, ρ, cs) ≈ ρ * cs^2

    # NaN guard
    @test isnan(Pressure(Isothermal, -1.0, cs))
end

@testset "Pressure — LocallyIsothermal" begin
    ρ = 1.0;  r = 2.0;  cs0 = 1.0;  q = 0.25
    P = Pressure(LocallyIsothermal, ρ, r, cs0, q)
    @test P ≈ ρ * (cs0 * r^(-q))^2

    # NaN guards
    @test isnan(Pressure(LocallyIsothermal, -1.0, r, cs0, q))
    @test isnan(Pressure(LocallyIsothermal, ρ, 0.0, cs0, q))
end

# ── 2a. Coordinate transforms — scalar round-trip ───────────────────── #

@testset "Coordinate transform — cart ↔ cylin round-trip" begin
    # 2D
    x, y = 3.0, 4.0
    s, ϕ = _cart2cylin(x, y)
    @test s ≈ 5.0
    @test 0 ≤ ϕ < 2π
    x2, y2 = _cylin2cart(s, ϕ)
    @test x2 ≈ x  atol = 1e-14
    @test y2 ≈ y  atol = 1e-14

    # 3D
    z = 7.0
    s3, ϕ3, z3 = _cart2cylin(x, y, z)
    @test z3 ≈ z
    x3, y3, z3b = _cylin2cart(s3, ϕ3, z3)
    @test x3 ≈ x  atol = 1e-14
    @test y3 ≈ y  atol = 1e-14
    @test z3b ≈ z  atol = 1e-14

    # Negative quadrant
    xn, yn = -1.0, -1.0
    sn, ϕn = _cart2cylin(xn, yn)
    @test sn ≈ sqrt(2.0)
    @test π < ϕn < 1.5π  # third quadrant
    xn2, yn2 = _cylin2cart(sn, ϕn)
    @test xn2 ≈ xn  atol = 1e-14
    @test yn2 ≈ yn  atol = 1e-14

    # Origin
    s0, ϕ0 = _cart2cylin(0.0, 0.0)
    @test s0 ≈ 0.0
end

# ── 2b. Vector field coordinate transforms ───────────────────────────── #

@testset "Coordinate transform — vector cart ↔ cylin round-trip" begin
    ϕ = π / 3
    Ax, Ay = 1.0, 2.0
    As, Aϕ = _vector_cart2cylin(ϕ, Ax, Ay)
    Ax2, Ay2 = _vector_cylin2cart(ϕ, As, Aϕ)
    @test Ax2 ≈ Ax  atol = 1e-14
    @test Ay2 ≈ Ay  atol = 1e-14

    # 3D with Az pass-through
    Az = 5.0
    As3, Aϕ3, Az3 = _vector_cart2cylin(ϕ, Ax, Ay, Az)
    Ax3, Ay3, Az3b = _vector_cylin2cart(ϕ, As3, Aϕ3, Az3)
    @test Az3 ≈ Az
    @test Ax3 ≈ Ax  atol = 1e-14
    @test Ay3 ≈ Ay  atol = 1e-14
end

# ── 3a. meshgrid ─────────────────────────────────────────────────────── #

@testset "meshgrid — 2D and 3D" begin
    xs = [1.0, 2.0, 3.0]
    ys = [10.0, 20.0]

    X, Y = meshgrid(xs, ys)
    @test size(X) == (3, 2)
    @test size(Y) == (3, 2)
    @test X[:, 1] == xs
    @test X[:, 2] == xs
    @test Y[1, :] == ys
    @test Y[3, :] == ys

    # 3D
    zs = [100.0]
    X3, Y3, Z3 = meshgrid(xs, ys, zs)
    @test size(X3) == (3, 2, 1)
end

# ── 3b. NaN-safe statistics ──────────────────────────────────────────── #

@testset "NaN-safe statistics" begin
    A = [1.0, NaN, 3.0, NaN, 5.0]

    @test nanmean(A) ≈ 3.0
    @test nanmaximum(A) ≈ 5.0
    @test nanminimum(A) ≈ 1.0

    # All NaN → NaN
    @test isnan(nanmean([NaN, NaN]))

    # No NaN → same as standard
    B = [2.0, 4.0, 6.0]
    @test nanmean(B) ≈ 4.0
    @test nanstd(B) ≈ 2.0  # population std: √(8/3) ≈ 1.63, sample std = 2.0
end

# ── 3c. Euclidean distance ───────────────────────────────────────────── #

@testset "Euclidean distance" begin
    x = [0.0, 3.0];  y = [0.0, 4.0];  z = [0.0, 0.0]

    # 3D
    d3 = Euclidean_distance(x, y, z, (0.0, 0.0, 0.0))
    @test d3[1] ≈ 0.0
    @test d3[2] ≈ 5.0

    # 2D
    d2 = Euclidean_distance(x, y, (0.0, 0.0))
    @test d2[1] ≈ 0.0
    @test d2[2] ≈ 5.0
end

# ── 3d. invert_order ─────────────────────────────────────────────────── #

@testset "invert_order — inverse permutation identity" begin
    order = [3, 1, 4, 2]
    inv_order = invert_order(order)

    # Applying the permutation then inverse permutation recovers identity
    v = [10, 20, 30, 40]
    @test v[order][inv_order] == v
end

# ── 3e. maxabs ───────────────────────────────────────────────────────── #

@testset "maxabs — complex array" begin
    A = ComplexF64[1+2im, 3+4im, -5+0im]
    @test maxabs(A) ≈ 5.0  # max |z| = |-5+0im| = 5
end
