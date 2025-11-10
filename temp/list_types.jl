using PhantomRevealer

const IO = PhantomRevealer.IO

prdfs = IO.read_phantom(joinpath("test", "testinput", "testdumpfile_00000"); separate_types = :all)
println("count: ", length(prdfs))
println("types: ", [get(prdf.params, :itype, missing) for prdf in prdfs])
println("npart: ", [IO.get_npart(prdf) for prdf in prdfs])
