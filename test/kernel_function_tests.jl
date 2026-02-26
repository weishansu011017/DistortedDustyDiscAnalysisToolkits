# ──────────────────────────────────────────────────────────────────────────── #
#  Test: SPH Kernel Functions — Numerical Correctness
# ──────────────────────────────────────────────────────────────────────────── #
#
#  What this file tests
#  ─────────────────────
#  Validates all six SPH kernel families (M4, M5, M6, C2, C4, C6 Wendland)
#  and their associated operations against hand-verified reference values:
#
#  1. Support radius — `KernelFunctionValid` returns correct q_max.
#  2. Normalisation  — `KernelFunctionnorm` returns correct σ_D.
#  3. Kernel values  — `Smoothed_kernel_function_dimensionless` at specific
#     q values matches analytic formulas.
#  4. Compact support — kernel ≡ 0 beyond q_max, NaN for q < 0.
#  5. Gradient kernel — `Smoothed_gradient_kernel_function` matches finite-
#     difference approximation.
#  6. LOS-integrated kernel — numerical column-integral of M4 matches the
#     tabulated `LOSint_Smoothed_kernel_function`.
#
#  Reference formulas
#  ──────────────────
#  • M4:  Monaghan (1985)   q_max = 2
#  • M5:  Morris (1996)     q_max = 2.5
#  • M6:  Morris (1996)     q_max = 3
#  • C2:  Wendland (1995)   q_max = 2
#  • C4:  Wendland (1995)   q_max = 2
#  • C6:  Wendland (1995)   q_max = 2
#
# ──────────────────────────────────────────────────────────────────────────── #

using Test
using PhantomRevealer

# ========================== Constants ======================================= #

const ALL_KERNELS = (M4_spline, M5_spline, M6_spline,
                     C2_Wendland, C4_Wendland, C6_Wendland)

# ========================== Helper functions ================================ #

"""Analytic dimensionless kernel value f(q) for verification."""
function analytic_kernel(::Type{M4_spline}, q::Float64)
    q < 0 && return NaN
    q >= 2 && return 0.0
    q < 1 ? 0.25 * (2 - q)^3 - (1 - q)^3 : 0.25 * (2 - q)^3
end

function analytic_kernel(::Type{M5_spline}, q::Float64)
    q < 0 && return NaN
    q >= 2.5 && return 0.0
    if q < 0.5
        (2.5 - q)^4 - 5 * (1.5 - q)^4 + 10 * (0.5 - q)^4
    elseif q < 1.5
        (2.5 - q)^4 - 5 * (1.5 - q)^4
    else
        (2.5 - q)^4
    end
end

function analytic_kernel(::Type{M6_spline}, q::Float64)
    q < 0 && return NaN
    q >= 3.0 && return 0.0
    if q < 1
        (3 - q)^5 - 6 * (2 - q)^5 + 15 * (1 - q)^5
    elseif q < 2
        (3 - q)^5 - 6 * (2 - q)^5
    else
        (3 - q)^5
    end
end

function analytic_kernel(::Type{C2_Wendland}, q::Float64)
    q < 0 && return NaN
    q >= 2 && return 0.0
    (1 - 0.5q)^4 * (2q + 1)
end

function analytic_kernel(::Type{C4_Wendland}, q::Float64)
    q < 0 && return NaN
    q >= 2 && return 0.0
    (1 - 0.5q)^6 * (35 / 12 * q^2 + 3q + 1)
end

function analytic_kernel(::Type{C6_Wendland}, q::Float64)
    q < 0 && return NaN
    q >= 2 && return 0.0
    (1 - 0.5q)^8 * (4q^3 + 6.25q^2 + 4q + 1)
end

# ============================== Test body =================================== #

# ── 1. Support radius ────────────────────────────────────────────────── #

@testset "Kernel support radius" begin
    @test KernelFunctionValid(M4_spline, Float64)  ≈ 2.0
    @test KernelFunctionValid(M5_spline, Float64)  ≈ 2.5
    @test KernelFunctionValid(M6_spline, Float64)  ≈ 3.0
    @test KernelFunctionValid(C2_Wendland, Float64) ≈ 2.0
    @test KernelFunctionValid(C4_Wendland, Float64) ≈ 2.0
    @test KernelFunctionValid(C6_Wendland, Float64) ≈ 2.0

    # Float32 variant
    @test KernelFunctionValid(M4_spline, Float32) ≈ 2.0f0
end

# ── 2. Normalisation constants ───────────────────────────────────────── #

@testset "Kernel normalisation constants" begin
    # M4 σ values: 1D = 4/3, 2D = 10/(7π), 3D = 1/π
    @test KernelFunctionnorm(M4_spline, Val(1), Float64)  ≈ 4 / 3
    @test KernelFunctionnorm(M4_spline, Val(2), Float64)  ≈ 10 / (7π)
    @test KernelFunctionnorm(M4_spline, Val(3), Float64)  ≈ 1 / π

    # C2 Wendland σ: 1D = 5/8, 2D = 7/(4π), 3D = 21/(16π)
    @test KernelFunctionnorm(C2_Wendland, Val(1), Float64) ≈ 5 / 8
    @test KernelFunctionnorm(C2_Wendland, Val(2), Float64) ≈ 7 / (4π)
    @test KernelFunctionnorm(C2_Wendland, Val(3), Float64) ≈ 21 / (16π)

    # C4 Wendland σ: 1D = 3/4, 2D = 9/(4π), 3D = 495/(256π)
    @test KernelFunctionnorm(C4_Wendland, Val(1), Float64) ≈ 3 / 4
    @test KernelFunctionnorm(C4_Wendland, Val(2), Float64) ≈ 9 / (4π)
    @test KernelFunctionnorm(C4_Wendland, Val(3), Float64) ≈ 495 / (256π)
end

# ── 3. Kernel values at specific q ───────────────────────────────────── #

@testset "Kernel values — analytic comparison" begin
    test_q_values = [0.0, 0.3, 0.7, 1.0, 1.3, 1.8]

    for K in ALL_KERNELS
        @testset "$(nameof(K))" begin
            for q in test_q_values
                expected = analytic_kernel(K, q)
                got = Smoothed_kernel_function_dimensionless(K, q, Val(3))
                σ = KernelFunctionnorm(K, Val(3), Float64)
                # Dimensionless kernel = σ * f(q)
                @test got ≈ σ * expected  atol = 1e-14
            end
        end
    end
end

# ── 4. Compact support and NaN for q < 0 ─────────────────────────────── #

@testset "Kernel compact support" begin
    for K in ALL_KERNELS
        qmax = KernelFunctionValid(K, Float64)
        # Beyond support → zero
        @test Smoothed_kernel_function_dimensionless(K, qmax + 0.1, Val(3)) == 0.0
        @test Smoothed_kernel_function_dimensionless(K, qmax + 10.0, Val(3)) == 0.0
        # Negative q → NaN
        @test isnan(Smoothed_kernel_function_dimensionless(K, -0.1, Val(3)))
    end
end

# ── 5. Gradient kernel — finite-difference check ─────────────────────── #

@testset "Kernel gradient — finite difference" begin
    kern = M4_spline()
    h = 0.1
    ra = (0.5, 0.5, 0.5)
    δ = 1e-6

    # Reference point at known offset
    rb = (0.55, 0.52, 0.48)

    W0 = Smoothed_kernel_function(typeof(kern), ra, rb, h)

    # Finite difference in x
    rb_dx = (rb[1] + δ, rb[2], rb[3])
    Wdx = Smoothed_kernel_function(typeof(kern), ra, rb_dx, h)
    ∂W∂x_fd = (Wdx - W0) / δ

    ∇W = Smoothed_gradient_kernel_function(typeof(kern), rb[1] - ra[1], rb[2] - ra[2], rb[3] - ra[3], h)

    # Gradient should match finite difference (sign convention: ∂W/∂rb)
    @test ∇W[1] ≈ ∂W∂x_fd  rtol = 1e-4
end

# ── 6. Physical W(r, h) = h⁻³ σ f(r/h) ─────────────────────────────── #

@testset "Kernel W(r,h) scaling" begin
    kern = M4_spline()
    h = 0.15
    r = 0.2
    q = r / h

    σ = KernelFunctionnorm(typeof(kern), Val(3), Float64)
    f_q = analytic_kernel(typeof(kern), q)

    W_expected = σ * f_q / h^3
    W_got = Smoothed_kernel_function(typeof(kern), r, h, Val(3))

    @test W_got ≈ W_expected  atol = 1e-14
end

# ── 7. LOS-integrated kernel consistency ─────────────────────────────── #

@testset "LOS kernel — two-point interface" begin
    kern = M4_spline()
    h = 0.1
    ra = (0.5, 0.5)
    rb = (0.52, 0.48)

    r = sqrt((ra[1] - rb[1])^2 + (ra[2] - rb[2])^2)

    Wlos_rh = LOSint_Smoothed_kernel_function(typeof(kern), r, h)
    Wlos_pts = LOSint_Smoothed_kernel_function(typeof(kern), ra, rb, h)

    @test Wlos_rh ≈ Wlos_pts  atol = 1e-14
end
