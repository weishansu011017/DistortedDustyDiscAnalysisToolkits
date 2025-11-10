using Test
using DataFrames

using PhantomRevealer

const IO = PhantomRevealer.IO
const KI = PhantomRevealer.KernelInterpolation
const NS = PhantomRevealer.NeighborSearch
const MassFromColumn = PhantomRevealer.MassFromColumn

function _neighbor_selection(pool, stack, lbvh, strategy, point, multiplier, h_values, gather_radius)
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
        throw(ArgumentError("Unknown strategy: $(strategy)"))
    end
end

@testset "GeneralGridInterpolation reproduces single-point kernels" begin
    df = DataFrame(
        x = [0.0, 0.2, 0.4, 0.6],
        y = [0.0, 0.1, -0.1, 0.3],
        z = [0.0, 0.0, 0.2, -0.2],
        h = [0.18, 0.22, 0.16, 0.24],
        rho = [1.0, 0.9, 1.1, 1.05],
        mass = [1.0, 0.8, 1.2, 0.9],
        P = [10.0, 12.0, 11.0, 13.0],
        vx = [0.2, -0.1, 0.0, 0.1],
        vy = [0.0, 0.05, -0.05, 0.02],
        vz = [-0.1, 0.0, 0.1, -0.02],
        Bx = [1.0, 1.5, 1.2, 0.8],
        By = [0.5, 0.6, 0.7, 0.9],
        Bz = [1.1, 0.9, 1.4, 1.3],
    )
    params = Dict{Symbol, Any}()
    prdf = IO.PhantomRevealerDataFrame(df, params)
    mass_source = MassFromColumn(:mass)

    input_base, catalog = PhantomRevealer.build_input(
        prdf,
        mass_source;
        scalars = [:P],
        gradients = [:P],
        divergences = [:v],
        curls = [:B],
    )

    coords = NTuple{3, Float64}[(0.05, 0.05, 0.0), (0.35, 0.05, 0.1)]
    grid_template = KI.GeneralGrid{3, Float64, Vector{Float64}, Vector{NTuple{3, Float64}}}(
        zeros(Float64, length(coords)),
        Vector{NTuple{3, Float64}}(coords),
    )

    strategies = (KI.itpSymmetric, KI.itpScatter, KI.itpGather)
    scalar_count = length(catalog.scalar_slots)

    for strategy in strategies
        input_expected = deepcopy(input_base)
        lbvh = KI.LinearBVH!(input_expected, CodeType = UInt64)
        multiplier = KI.KernelFunctionValid(input_expected.smoothed_kernel, eltype(input_expected.h))
        mean_h = isempty(input_expected.h) ? zero(eltype(input_expected.h)) : sum(input_expected.h) / length(input_expected.h)
        gather_radius = multiplier * mean_h
        pool_length = max(1, length(input_expected.x))
        stack_length = max(1, 2 * length(lbvh.brt.left_child) + 8)

        expected_values = Vector{Vector{Float64}}(undef, length(coords))
        for (pt_idx, point) in enumerate(coords)
            pool = zeros(Int, pool_length)
            stack = Vector{Int}(undef, stack_length)
            selection, ha = _neighbor_selection(pool, stack, lbvh, strategy, point, multiplier, input_expected.h, gather_radius)

            values = Float64[]
            if scalar_count > 0
                workspace = zeros(eltype(input_expected.x), scalar_count)
                KI.quantities_interpolate!(workspace, input_expected, point, ha, selection, catalog.scalar_slots, strategy)
                append!(values, workspace)
            end

            for slot in catalog.grad_slots
                grad = slot == 0 ? KI.gradient_density(input_expected, point, ha, selection, strategy) : KI.gradient_quantity_interpolate(input_expected, point, ha, selection, slot, strategy)
                append!(values, Tuple(grad))
            end

            for slot in catalog.div_slots
                ax, ay, az = slot
                div_val = KI.divergence_quantity_interpolate(input_expected, point, ha, selection, ax, ay, az, strategy)
                push!(values, div_val)
            end

            for slot in catalog.curl_slots
                ax, ay, az = slot
                curl = KI.curl_quantity_interpolate(input_expected, point, ha, selection, ax, ay, az, strategy)
                append!(values, Tuple(curl))
            end

            expected_values[pt_idx] = values
        end

        input_grid = deepcopy(input_base)
        grids, order = KI.GeneralGridInterpolation(grid_template, input_grid, catalog, strategy)

        @test order == catalog.ordered_names
        @test length(grids) == length(order)

        for g in grids
            @test g.coor === grid_template.coor
        end

        for (pt_idx, _) in enumerate(coords)
            actual = [grids[k].grid[pt_idx] for k in 1:length(grids)]
            expected = expected_values[pt_idx]
            @test actual ≈ expected rtol = 1e-5 atol = 1e-7
        end
    end
end

@testset "GeneralGridInterpolation matches dumpfile divergence" begin
    prdfs = IO.read_phantom(joinpath(@__DIR__, "testinput", "testdumpfile_00000"); separate_types = :all)
    gas_candidates = filter(prdf -> IO.get_npart(prdf) > 0 && get(prdf.params, :itype, 1) == 1, prdfs)
    @test !isempty(gas_candidates)

    gas_full = first(gas_candidates)
    sample_count = min(512, IO.get_npart(gas_full))
    gas = IO.PhantomRevealerDataFrame(DataFrame(gas_full.dfdata[1:sample_count, :]), deepcopy(gas_full.params))
    IO.add_rho!(gas)

    mass_source = PhantomRevealer.MassFromParams(:mass)
    input_base, catalog = PhantomRevealer.build_input(
        gas,
        mass_source;
        scalars = Symbol[],
        gradients = Symbol[],
        divergences = [:v],
        curls = Symbol[],
    )

    grid_template = KI.GeneralGrid(gas.dfdata.x, gas.dfdata.y, gas.dfdata.z)
    divv_column = Float64.(gas.dfdata.divv)
    strategies = (KI.itpSymmetric, KI.itpScatter, KI.itpGather)

    for strategy in strategies
        input_grid = deepcopy(input_base)
        grids, order = KI.GeneralGridInterpolation(grid_template, input_grid, catalog, strategy)
        idx = findfirst(==(Symbol("∇⋅v")), order)
        @test idx !== nothing

        interpolated = grids[idx].grid
        @test !any(isnan, interpolated)
        @test !all(iszero, interpolated)

        reference = convert.(eltype(interpolated), divv_column)
        @test interpolated ≈ reference rtol = 5e-3 atol = 1e-6
    end
end
