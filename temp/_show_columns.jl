using PhantomRevealer

path = joinpath(@__DIR__, "..", "test", "testinput", "testdumpfile_00000")
prdfs = read_phantom(path; separate_types = :all)
println("Total frames: ", length(prdfs))
for (idx, prdf) in enumerate(prdfs)
    println("frame ", idx, " itype=", get(prdf.params, :itype, missing), " npart=", PhantomRevealer.IO.get_npart(prdf))
    println(names(prdf.dfdata))
end
