using Partia
prdf_list = read_phantom("./test/testinput/testdumpfile_00000", separate_types=:all);
datag = prdf_list[1];
add_rho!(datag);
input, catalog = build_input(datag, MassFromParams(:mass), scalars=Symbol[:rho],divergences=[:v],  smoothed_kernel=M6_spline);
grid = GeneralGrid(datag.dfdata.x, datag.dfdata.y, datag.dfdata.z);
generalgridresult = GeneralGridInterpolation(grid, input, catalog, itpGather);