using PhantomRevealer

const IO = PhantomRevealer.IO
const KI = PhantomRevealer.KernelInterpolation
const NS = PhantomRevealer.NeighborSearch
const MassFromParams = PhantomRevealer.MassFromParams

function neighbor_selection(pool, stack, lbvh, strategy, point, multiplier, h_values, gather_radius)
    if strategy == KI.itpSymmetric || strategy == KI.itpScatter
        return NS.LBVH_query!(pool, lbvh, point, multiplier, h_values)
    elseif strategy == KI.itpGather
        selection = NS.LBVH_query!(pool, lbvh, point, gather_radius)
        if selection.count == 0
            return selection, zero(eltype(h_values))
        end
        nearest = NS.nearest_index(selection)
        ha = h_values[nearest]
        radius = multiplier * ha
        selection = NS.LBVH_query!(pool, lbvh, point, radius)
        nearest = selection.count == 0 ? nearest : NS.nearest_index(selection)
        return selection, h_values[nearest]
    else
        error("Unknown strategy $strategy")
    end
end

function grad_for_strategy(input, strategy, point, neighbor_index, ha)
    kernel = input.smoothed_kernel
    rb = (input.x[neighbor_index], input.y[neighbor_index], input.z[neighbor_index])
    if strategy == KI.itpGather
        return KI.Smoothed_gradient_kernel_function(kernel, point, rb, ha)
    elseif strategy == KI.itpScatter
        return KI.Smoothed_gradient_kernel_function(kernel, point, rb, input.h[neighbor_index])
    else
        ∇Wa = KI.Smoothed_gradient_kernel_function(kernel, point, rb, ha)
        ∇Wb = KI.Smoothed_gradient_kernel_function(kernel, point, rb, input.h[neighbor_index])
        return (0.5 * (∇Wa[1] + ∇Wb[1]), 0.5 * (∇Wa[2] + ∇Wb[2]), 0.5 * (∇Wa[3] + ∇Wb[3]))
    end
end

function dot3(a, b)
    return a[1]*b[1] + a[2]*b[2] + a[3]*b[3]
end

function divergence_candidate(input, div_slot, idx, selection, ha, strategy, weight_mode, divide_mode)
    va = (
        input.quant[div_slot[1]][idx],
        input.quant[div_slot[2]][idx],
        input.quant[div_slot[3]][idx],
    )
    rho_a = input.ρ[idx]
    total = zero(rho_a)
    for k in 1:selection.count
        j = selection.pool[k]
        mb = input.m[j]
        rho_b = input.ρ[j]
        vb = (
            input.quant[div_slot[1]][j],
            input.quant[div_slot[2]][j],
            input.quant[div_slot[3]][j],
        )
        dv = (vb[1] - va[1], vb[2] - va[2], vb[3] - va[3])
        grad = grad_for_strategy(input, strategy, (input.x[idx], input.y[idx], input.z[idx]), j, ha)
        base = dot3(dv, grad)
        weight = if weight_mode == :over_rhob
            mb / rho_b
        elseif weight_mode == :over_rhoa
            mb / rho_a
        elseif weight_mode == :plain
            mb
        else
            error("Unknown weight mode $weight_mode")
        end
        total += weight * base
    end
    if divide_mode == :over_rhoa
        return total / rho_a
    elseif divide_mode == :none
        return total
    else
        error("Unknown divide mode $divide_mode")
    end
end

function main()
    dump_path = joinpath(@__DIR__, "..", "test", "testinput", "testdumpfile_00000")
    prdfs = IO.read_phantom(dump_path; separate_types = :all)
    gas_candidates = filter(prdf -> IO.get_npart(prdf) > 0 && get(prdf.params, :itype, 1) == 1, prdfs)
    @assert !isempty(gas_candidates)
    gas_data = first(gas_candidates)
    IO.add_rho!(gas_data)

    input, catalog = PhantomRevealer.build_input(
        gas_data,
        MassFromParams(:mass);
        scalars = Symbol[],
        gradients = Symbol[],
        divergences = [:v],
        curls = Symbol[],
    )

    lbvh = KI.LinearBVH!(input, Val(3))
    multiplier = KI.KernelFunctionValid(input.smoothed_kernel, eltype(input.h))
    mean_h = isempty(input.h) ? zero(eltype(input.h)) : sum(input.h) / length(input.h)
    gather_radius = multiplier * mean_h
    pool = zeros(Int, length(input.x))
    stack = Vector{Int}(undef, max(1, 2 * length(lbvh.brt.left_child) + 8))

    strategies = (KI.itpSymmetric, KI.itpScatter, KI.itpGather)
    variants = (
        (:over_rhob, :none, :"mb/ρb"),
        (:over_rhob, :over_rhoa, :"mb/ρb ÷ ρa"),
        (:over_rhoa, :none, :"mb/ρa"),
        (:over_rhoa, :over_rhoa, :"mb/ρa ÷ ρa"),
        (:plain, :none, :"mb"),
        (:plain, :over_rhoa, :"mb ÷ ρa"),
    )
    sample_step = max(1, fld(input.Npart, 64))
    sample_indices = collect(1:sample_step:input.Npart)
    println("Comparing divergences for $(length(sample_indices)) samples")
    for strategy in strategies
        println("Strategy: $strategy")
        for idx in sample_indices
            point = (input.x[idx], input.y[idx], input.z[idx])
            selection, ha = neighbor_selection(pool, stack, lbvh, strategy, point, multiplier, input.h, gather_radius)
            if selection.count == 0
                continue
            end
            pr_reference = gas_data.dfdata.divv[idx]
            println("idx=$idx divv=$(pr_reference)")
            for (weight_mode, divide_mode, label) in variants
                val = divergence_candidate(input, catalog.div_slots[1], idx, selection, ha, strategy, weight_mode, divide_mode)
                println("  $label => $val")
            end
        end
    end
end

main()
