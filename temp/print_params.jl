using PhantomRevealer

const IO = PhantomRevealer.IO

prdfs = IO.read_phantom(joinpath("test", "testinput", "testdumpfile_00000"); separate_types = :all)
for prdf in prdfs
    println("itype = ", get(prdf.params, :itype, missing))
    println(prdf.params)
    println()
end
