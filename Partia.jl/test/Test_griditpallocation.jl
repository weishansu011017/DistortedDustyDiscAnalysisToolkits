using Partia
using BenchmarkTools

pdf_list = pdf_list = read_phantom(joinpath(@__DIR__, "testinput","testdumpfile_00000"), separate_types=:all);
datag = pdf_list[1];
add_rho!(datag);
input, catalog = build_input(datag, MassFromParams(:mass), scalars=Symbol[:rho, :vx, :vy, :vz], gradients = [:rho] ,divergences=[:v], curls=[:v],  smoothed_kernel=M6_spline);
grid = GeneralGrid(datag.dfdata.x, datag.dfdata.y, datag.dfdata.z);
LBVH = LinearBVH!(input, Val(3), CodeType=UInt64);
catalog_concise = to_concise_catalog(catalog);

@info "----------------------------- DOING GRID INTERPOLATION TESTS -----------------------------"
griditpresultga = GeneralGrid_interpolation(CPUComputeBackend(), grid, input, catalog, itpGather);
println()
griditpresultga = GeneralGrid_interpolation(CPUComputeBackend(), grid, input, catalog, itpGather);
@info "--------------------------------- GRID INTERPOLATION DONE ---------------------------------"
griditpresultsc = GeneralGrid_interpolation(CPUComputeBackend(), grid, input, catalog, itpScatter);
griditpresultsy = GeneralGrid_interpolation(CPUComputeBackend(), grid, input, catalog, itpSymmetric);

@btime KernelInterpolation._general_quantity_interpolate_kernel($input, (40.0,40.0,0.0), 2.0, $LBVH, $catalog_concise,itpGather);