using PhantomRevealer
using DataFrames
const IO = PhantomRevealer.IO
const KI = PhantomRevealer.KernelInterpolation
const NS = PhantomRevealer.NeighborSearch

prdfs = IO.read_phantom(joinpath("test", "testinput", "testdumpfile_00000"); separate_types = :all)
gas_candidates = filter(prdf -> IO.get_npart(prdf) > 0 && get(prdf.params, :itype, 1) == 1, prdfs)
@assert !isempty(gas_candidates)

gas_full = first(gas_candidates)

gas = IO.PhantomRevealerDataFrame(DataFrame(gas_full.dfdata[1:128, :]), deepcopy(gas_full.params))
IO.add_rho!(gas)
println("mass param: ", gas.params[:mass])
println("rho[1]: ", gas.dfdata.rho[1])

mass_source = PhantomRevealer.MassFromParams(:mass)
input, catalog = PhantomRevealer.build_input(gas, mass_source; scalars = Symbol[], gradients = Symbol[], divergences = [:v], curls = Symbol[])

grid = KI.GeneralGrid(gas.dfdata.x, gas.dfdata.y, gas.dfdata.z)

LBVH = KI.LinearBVH!(input, CodeType = UInt64)
pool = zeros(Int, length(input.x))
stack = Vector{Int}(undef, max(1, 2 * length(LBVH.brt.left_child) + 8))
multiplier = KI.KernelFunctionValid(input.smoothed_kernel, eltype(input.h))

point = grid.coor[1]
selection_sym, ha_sym = NS.LBVH_query!(pool, stack, LBVH, point, multiplier, input.h)
println("Symmetric selection count: ", selection_sym.count, " ha=", ha_sym)
div_slots = catalog.div_slots[1]
div_manual = KI._divergence_quantity_interpolate_kernel(input, point, ha_sym, selection_sym, div_slots[1], div_slots[2], div_slots[3], KI.itpSymmetric)
println("Manual divergence: ", div_manual)

if selection_sym.count > 0
	idx1 = selection_sym.pool[1]
	mb = input.m[idx1]
	ρb = input.ρ[idx1]
	rb = (input.x[idx1], input.y[idx1], input.z[idx1])
	W_ha = KI.Smoothed_kernel_function(input.smoothed_kernel, point, rb, ha_sym)
	W_hb = KI.Smoothed_kernel_function(input.smoothed_kernel, point, rb, input.h[idx1])
	W_sym = 0.5 * (W_ha + W_hb)
	println("First neighbor -> mb=", mb, " rho_b=", ρb, " W_ha=", W_ha, " W_hb=", W_hb, " W_sym=", W_sym)

	∂W = KI.Smoothed_gradient_kernel_function(input.smoothed_kernel, point, rb, ha_sym)
	∂Wb = KI.Smoothed_gradient_kernel_function(input.smoothed_kernel, point, rb, input.h[idx1])
	grad_sym = 0.5 .* (∂W .+ ∂Wb)
	println("grad_sym=", grad_sym)

	Ax_b = input.quant[div_slots[1]][idx1]
	Ay_b = input.quant[div_slots[2]][idx1]
	Az_b = input.quant[div_slots[3]][idx1]

	mbW = mb * W_sym
	mWlρ = mbW / ρb
	ρ_val = mbW
	Ax_val = mbW * Ax_b / ρb
	Ay_val = mbW * Ay_b / ρb
	Az_val = mbW * Az_b / ρb
	∇Af_val = mb * (grad_sym[1] * Ax_b + grad_sym[2] * Ay_b + grad_sym[3] * Az_b)
	∇Axb_val = mb * grad_sym[1]
	∇Ayb_val = mb * grad_sym[2]
	∇Azb_val = mb * grad_sym[3]
	println("mWlρ=", mWlρ, " rho=", ρ_val, " Ax_val=", Ax_val, " Ay_val=", Ay_val, " Az_val=", Az_val)
	println("∇Af=", ∇Af_val, " ∇Axb=", ∇Axb_val, " ∇Ayb=", ∇Ayb_val, " ∇Azb=", ∇Azb_val)
end

gather_radius = multiplier * (sum(input.h) / length(input.h))
selection_gather = NS.LBVH_query!(pool, stack, LBVH, point, gather_radius)
println("Gather initial count: ", selection_gather.count)

grids_sym, order_sym = KI.GeneralGridInterpolation(grid, input, catalog, KI.itpSymmetric)
idx = findfirst(==(Symbol("∇⋅v")), order_sym)
println("Sym first 5: ", grids_sym[idx].grid[1:5])

grids_gather, order_gather = KI.GeneralGridInterpolation(grid, input, catalog, KI.itpGather)
idxg = findfirst(==(Symbol("∇⋅v")), order_gather)
println("Gather first 5: ", grids_gather[idxg].grid[1:5])
