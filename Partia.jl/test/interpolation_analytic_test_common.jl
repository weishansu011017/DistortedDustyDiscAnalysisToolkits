using Random
using Partia

"""Shared analytic manufactured-field helpers for interpolation regression tests."""

@inline analytic_scalar(x, y, z) = x + y + z
@inline analytic_grad_scalar(::Float64, ::Float64, ::Float64) = (1.0, 1.0, 1.0)
@inline analytic_vecA(x, y, z) = (x, y, z)
@inline analytic_divA(::Float64, ::Float64, ::Float64) = 3.0
@inline analytic_curlA(::Float64, ::Float64, ::Float64) = (0.0, 0.0, 0.0)

function make_uniform_cloud_3d(nx::Int; eta::Float64, kernel::Type{K} = M4_spline, variable_h::Bool = false) where {K<:AbstractSPHKernel}
    dx = 1.0 / nx
    coords = collect(range(dx / 2, stop = 1.0 - dx / 2, step = dx))
    x = Float64[]
    y = Float64[]
    z = Float64[]
    @inbounds for xi in coords, yi in coords, zi in coords
        push!(x, xi)
        push!(y, yi)
        push!(z, zi)
    end

    n = length(x)
    m = fill(dx^3, n)
    h = similar(x)
    rho = ones(Float64, n)

    q = similar(x)
    vx = similar(x)
    vy = similar(x)
    vz = similar(x)
    @inbounds for i in 1:n
        h[i] = if variable_h
            eta * dx * (1.0 + 0.18 * (x[i] - 0.5) - 0.10 * (y[i] - 0.5) + 0.08 * (z[i] - 0.5))
        else
            eta * dx
        end
        q[i] = analytic_scalar(x[i], y[i], z[i])
        vx[i], vy[i], vz[i] = analytic_vecA(x[i], y[i], z[i])
    end

    input = InterpolationInput((x, y, z), m, h, rho, (q, vx, vy, vz); smoothed_kernel = kernel)
    catalog = InterpolationCatalog(
        (:q, :vx, :vy, :vz), Val(3);
        scalars = (:q,),
        gradients = (:q,),
        divergences = (:v,),
        curls = (:v,),
    )
    return input, catalog, h[1]
end

function sample_reference_points(rng::AbstractRNG, n::Int, h::Float64; kernel::Type{K} = M4_spline) where {K<:AbstractSPHKernel}
    margin = 1.5 * KernelFunctionValid(kernel, Float64) * h
    lo = margin
    hi = 1.0 - margin
    refs = NTuple{3, Float64}[]
    @inbounds for _ in 1:n
        x = rand(rng) * (hi - lo) + lo
        y = rand(rng) * (hi - lo) + lo
        z = rand(rng) * (hi - lo) + lo
        push!(refs, (x, y, z))
    end
    return refs
end

mean_abs(v) = isempty(v) ? 0.0 : sum(abs, v) / length(v)
