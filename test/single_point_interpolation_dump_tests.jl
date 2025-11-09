using Test
using PhantomRevealer

const IO = PhantomRevealer.IO
const KI = PhantomRevealer.KernelInterpolation
const NS = PhantomRevealer.NeighborSearch
const MassFromParams = PhantomRevealer.MassFromParams

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

    input, catalog = build_input(
        gas_data,
        MassFromParams(:mass);
        scalars = target_scalars,
        gradients = Symbol[],
        divergences = Symbol[],
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
    pool = zeros(Int, input.Npart)
    stack = Vector{Int}(undef, max(1, 2 * length(lbvh.brt.left_child) + 8))
    multiplier = KI.KernelFunctionValid(input.smoothed_kernel, T)

    # Query around the same particle using the adaptive radius helper
    selection, ha_query = NS.LBVH_query!(pool, stack, lbvh, reference_point, multiplier, input.h)
    @test selection.count > 0

    nearest = NS.nearest_index(selection)
    single_selection = NS.NeighborSelection([nearest], 1, nearest)
    KI.quantities_interpolate!(workspace, input, reference_point, ha_query, single_selection, columns)
    for (j, slot) in pairs(columns)
        nearest_value = slot == 0 ? KI.density(input, reference_point, ha_query, single_selection) : input.quant[slot][nearest]
        @test isapprox(workspace[j], nearest_value; rtol = eps(T) * 16)
    end

    # Full neighbor set produces a finite Shepard-normalised value
    values = KI.quantities_interpolate(input, reference_point, ha_query, selection, columns)
    @test isapprox(values[1], KI.density(input, reference_point, ha_query, selection); rtol = eps(T) * 32)
    for slot in velocity_slots
        interp_value = KI.quantity_interpolate(input, reference_point, ha_query, selection, slot)
        idx = findfirst(==(slot), columns)
        @test !isnothing(idx)
        @test isapprox(values[idx], interp_value; rtol = eps(T) * 64)
    end
end
