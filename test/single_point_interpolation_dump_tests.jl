using Test
using Statistics
using PhantomRevealer

const IO = PhantomRevealer.IO
const KI = PhantomRevealer.KernelInterpolation
const NS = PhantomRevealer.NeighborSearch
const MassFromParams = PhantomRevealer.MassFromParams

function _neighbor_selection(pool, stack, lbvh, strategy, point, multiplier, h_values)
    if strategy == KI.itpSymmetric || strategy == KI.itpScatter
        multiplier *= 2.5
    elseif strategy == KI.itpGather
        multiplier *= 1.5
    else
        throw(ArgumentError("Unknown strategy: $(strategy)"))
    end
    ha = mean(h_values)
    gather_radius = ha * multiplier
    selection = NS.LBVH_query!(pool, stack, lbvh, point, gather_radius)
    nearest = NS.nearest_index(selection)
    ha = h_values[nearest]
    radius = multiplier * ha
    selection = NS.LBVH_query!(pool, stack, lbvh, point, radius)
    nearest = selection.count == 0 ? nearest : NS.nearest_index(selection)
    return selection, h_values[nearest]
end

@testset "Single-point interpolation on dumpfile" begin
    dump_path = joinpath(@__DIR__, "testinput", "testdumpfile_00000")
    prdfs = read_phantom(dump_path; separate_types = :all)

    gas_candidates = filter(prdf -> IO.get_npart(prdf) > 0 && get(prdf.params, :itype, 1) == 1, prdfs)
    @test !isempty(gas_candidates)
    gas_data = gas_candidates[1]
    @test haskey(gas_data.params, :mass)
    @test haskey(gas_data.params, :hfact)

    IO.add_rho!(gas_data)
    target_scalars = [:rho, :vx, :vy, :vz]
    data_columns = Set(Symbol.(names(gas_data.dfdata)))
    @test all(name -> name in data_columns, target_scalars)

    divv_column = Float64.(gas_data.dfdata.divv)
    @test !all(iszero, divv_column)

    input, catalog = build_input(
        gas_data,
        MassFromParams(:mass);
        scalars = target_scalars,
        gradients = Symbol[],
        divergences = [:v],
        curls = Symbol[]
    )

    T = eltype(input.x)
    columns = catalog.scalar_slots
    density_slot = columns[1]
    velocity_slots = columns[2:end]

    # Ensure a single neighbor reproduces the particle value
    particle_index = 1
    reference_point = (input.x[particle_index], input.y[particle_index], input.z[particle_index])
    ha = input.h[particle_index]
    neighbors_single = NS.NeighborSelection([particle_index], 1, particle_index)

    workspace = zeros(T, length(columns))
    KI.quantities_interpolate!(workspace, input, reference_point, ha, neighbors_single, columns)
    for (j, slot) in pairs(columns)
        expected_value = slot == 0 ? KI.density(input, reference_point, ha, neighbors_single) : input.quant[slot][particle_index]
        @test isapprox(workspace[j], expected_value; rtol = eps(T) * 16)
    end

    # Build LBVH and query neighbors using the same dataset
    lbvh = KI.LinearBVH!(input)
    ordered_indices = Int.(lbvh.enc.order)
    divv_column = divv_column[ordered_indices]
    pool = zeros(Int, input.Npart)
    stack = Vector{Int}(undef, max(1, 2 * length(lbvh.brt.left_child) + 8))
    multiplier = KI.KernelFunctionValid(input.smoothed_kernel, T)

    # Divergence of velocity matches dumpfile column for sampled particles across strategies
    @test length(catalog.div_slots) == 1
    div_slot = catalog.div_slots[1]
    strategies = (KI.itpSymmetric, KI.itpScatter, KI.itpGather)

    multiplier = KI.KernelFunctionValid(input.smoothed_kernel, T)
    mean_h = isempty(input.h) ? zero(T) : sum(input.h) / length(input.h)
    gather_radius = multiplier * mean_h
    pool_length = max(1, length(input.x))
    stack_length = max(1, 2 * length(lbvh.brt.left_child) + 8)
    sample_step = max(1, fld(input.Npart, 256))
    sample_indices = collect(1:sample_step:input.Npart)

    for strategy in strategies
        pool = zeros(Int, pool_length)
        stack = Vector{Int}(undef, stack_length)
        for idx in sample_indices
            point = (input.x[idx], input.y[idx], input.z[idx])
            selection_s, ha_s = _neighbor_selection(pool, stack, lbvh, strategy, point, multiplier, input.h)
            @test selection_s.count > 0
            value = KI.divergence_quantity_interpolate(input, point, ha_s, selection_s, div_slot[1], div_slot[2], div_slot[3], strategy)
            @test isfinite(value)
            reference = convert(eltype(value), divv_column[idx])
            @test value ≈ reference rtol = 5e-1 atol = 1e-2
        end
    end
end
