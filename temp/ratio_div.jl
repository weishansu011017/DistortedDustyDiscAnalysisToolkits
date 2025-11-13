using PhantomRevealer
using Statistics

const IO = PhantomRevealer.IO
const KI = PhantomRevealer.KernelInterpolation
const NS = PhantomRevealer.NeighborSearch
const MassFromParams = PhantomRevealer.MassFromParams

function neighbor_selection(pool, stack, lbvh, strategy, point, multiplier, h_values, gather_radius)
    if strategy == KI.itpSymmetric || strategy == KI.itpScatter
        return NS.LBVH_query!(pool, stack, lbvh, point, multiplier, h_values)
    elseif strategy == KI.itpGather
        selection = NS.LBVH_query!(pool, stack, lbvh, point, gather_radius)
        if selection.count == 0
            return selection, zero(eltype(h_values))
        end
        nearest = NS.nearest_index(selection)
        ha = h_values[nearest]
        radius = multiplier * ha
        selection = NS.LBVH_query!(pool, stack, lbvh, point, radius)
        nearest = selection.count == 0 ? nearest : NS.nearest_index(selection)
        return selection, h_values[nearest]
    else
        error("Unknown strategy $strategy")
    end
end

function grad_symmetric(input, point, neighbor_index, ha)
    kernel = input.smoothed_kernel
    rb = (input.x[neighbor_index], input.y[neighbor_index], input.z[neighbor_index])
    ∇Wa = KI.Smoothed_gradient_kernel_function(kernel, point, rb, ha)
    ∇Wb = KI.Smoothed_gradient_kernel_function(kernel, point, rb, input.h[neighbor_index])
    return (0.5 * (∇Wa[1] + ∇Wb[1]), 0.5 * (∇Wa[2] + ∇Wb[2]), 0.5 * (∇Wa[3] + ∇Wb[3]))
end

function main()
    dump_path = joinpath(@__DIR__, "..", "test", "testinput", "testdumpfile_00000")
    prdfs = IO.read_phantom(dump_path; separate_types = :all)
    gas_candidates = filter(prdf -> IO.get_npart(prdf) > 0 && get(prdf.params, :itype, 1) == 1, prdfs)
    gas_data = first(gas_candidates)
    IO.add_rho!(gas_data)
    input, catalog = PhantomRevealer.build_input(
        gas_data,
        MassFromParams(:mass);
        scalars = Symbol[],
        gradients = Symbol[],
        divergences = [:v],
        curls = Symbol[],
        smoothed_kernel = KI.M4_spline,
    )

    lbvh = KI.LinearBVH!(input, Val(3))
    multiplier = KI.KernelFunctionValid(input.smoothed_kernel, eltype(input.h))
    mean_h = isempty(input.h) ? zero(eltype(input.h)) : sum(input.h) / length(input.h)
    gather_radius = multiplier * mean_h
    pool = zeros(Int, length(input.x))
    stack = Vector{Int}(undef, max(1, 2 * length(lbvh.brt.left_child) + 8))
    div_slot = catalog.div_slots[1]

    ratios = Float64[]
    diffs = Float64[]
    refs = Float64[]
    vals = Float64[]
    vals_rho = Float64[]
    vals_mWlρ = Float64[]
    vals_unscaled = Float64[]
    vals_norm = Float64[]

    step = max(1, fld(input.Npart, 2048))
    for idx in 1:step:input.Npart
        point = (input.x[idx], input.y[idx], input.z[idx])
        selection, ha = neighbor_selection(pool, stack, lbvh, KI.itpSymmetric, point, multiplier, input.h, gather_radius)
        if selection.count == 0
            continue
        end
    vx_a = input.quant[div_slot[1]][idx]
    vy_a = input.quant[div_slot[2]][idx]
    vz_a = input.quant[div_slot[3]][idx]
        val = 0.0
        val_rho = 0.0
        val_mWlρ = 0.0
        sum_grad = (0.0, 0.0, 0.0)
        sum_Agrad = 0.0
        sum_weighted_v = (0.0, 0.0, 0.0)
        for k in 1:selection.count
            j = selection.pool[k]
            mb = input.m[j]
            rho_b = input.ρ[j]
            grad = grad_symmetric(input, point, j, ha)
            W = 0.5 * (
                KI.Smoothed_kernel_function(input.smoothed_kernel, point, (input.x[j], input.y[j], input.z[j]), ha) +
                KI.Smoothed_kernel_function(input.smoothed_kernel, point, (input.x[j], input.y[j], input.z[j]), input.h[j])
            )
            val_rho += mb * W
            val_mWlρ += mb * W / rho_b
            weight = mb * W / rho_b
            sum_weighted_v = (
                sum_weighted_v[1] + weight * input.quant[div_slot[1]][j],
                sum_weighted_v[2] + weight * input.quant[div_slot[2]][j],
                sum_weighted_v[3] + weight * input.quant[div_slot[3]][j],
            )
            dvx = input.quant[div_slot[1]][j] - vx_a
            dvy = input.quant[div_slot[2]][j] - vy_a
            dvz = input.quant[div_slot[3]][j] - vz_a
            val += (mb / rho_b) * (dvx * grad[1] + dvy * grad[2] + dvz * grad[3])
            sum_grad = (
                sum_grad[1] + (mb / rho_b) * grad[1],
                sum_grad[2] + (mb / rho_b) * grad[2],
                sum_grad[3] + (mb / rho_b) * grad[3],
            )
            sum_Agrad += (mb / rho_b) * (
                input.quant[div_slot[1]][j] * grad[1] +
                input.quant[div_slot[2]][j] * grad[2] +
                input.quant[div_slot[3]][j] * grad[3]
            )
        end
        reference = gas_data.dfdata.divv[idx]
        push!(ratios, reference == 0 ? NaN : val / reference)
        push!(diffs, val - reference)
        push!(vals, val)
        push!(vals_unscaled, sum_Agrad - (vx_a * sum_grad[1] + vy_a * sum_grad[2] + vz_a * sum_grad[3]))
        Ax_norm = sum_weighted_v[1] / val_mWlρ
        Ay_norm = sum_weighted_v[2] / val_mWlρ
        Az_norm = sum_weighted_v[3] / val_mWlρ
        push!(vals_norm, sum_Agrad - (Ax_norm * sum_grad[1] + Ay_norm * sum_grad[2] + Az_norm * sum_grad[3]))
        push!(vals_rho, val_rho)
        push!(vals_mWlρ, val_mWlρ)
        push!(refs, reference)
    end

    ratios_clean = filter(!isnan, ratios)
    println("ratio stats: mean=$(mean(ratios_clean)), std=$(std(ratios_clean)), min=$(minimum(ratios_clean)), max=$(maximum(ratios_clean))")
    println("diff stats: mean=$(mean(diffs)), std=$(std(diffs)), min=$(minimum(diffs)), max=$(maximum(diffs))")
    println("ref stats: mean=$(mean(refs)), std=$(std(refs)), min=$(minimum(refs)), max=$(maximum(refs))")
    println("val stats: mean=$(mean(vals)), std=$(std(vals)), min=$(minimum(vals)), max=$(maximum(vals))")
    println("val_unscaled stats: mean=$(mean(vals_unscaled)), std=$(std(vals_unscaled))")
    println("val_norm stats: mean=$(mean(vals_norm)), std=$(std(vals_norm))")
    println("rho stats: mean=$(mean(vals_rho)), std=$(std(vals_rho))")
    println("mWlρ stats: mean=$(mean(vals_mWlρ)), std=$(std(vals_mWlρ))")
    α = cov(vals, refs) / var(vals)
    β = mean(refs) - α * mean(vals)
    println("fit: divv ≈ $(α) * val + $(β)")
    println("correlation = $(cor(vals, refs))")
end

main()
