# ParticleIO.jl

`ParticleIO.jl` is the particle-data front-end companion to `Partia.jl`. It provides `ParticleDataFrame`, Phantom dump-file reading, particle-side convenience operations, and the adapter layer that turns particle tables into `Partia` interpolation inputs.



## Installation

Before installing `ParticleIO.jl`, `Partia.jl` is required. Please install `Partia.jl` first.

`ParticleIO.jl` is not registered in the General registry. If you want to install it directly from this repository, use

```julia
using Pkg
Pkg.add(url="https://github.com/weishansu011017/DistortedDustyDiscAnalysisToolkits.git", subdir="ParticleIO.jl")
```

## Reading Phantom dump files

`ParticleIO.jl` currently provides a reader for the native binary dump format produced by `Phantom`. The main entry point is `read_phantom`, which returns a `Vector{ParticleDataFrame}`.

By default, SPH particles are returned in one `ParticleDataFrame` and sink particles in a second one:

```julia
using ParticleIO

data_list = read_phantom("dumpfile_00000")
gas, sinks = data_list
```

If a dump file contains multiple particle types, `separate_types = :all` splits them into separate particle tables:

```julia
data_list = read_phantom("dumpfile_00000"; separate_types = :all)
```

Inactive particles can also be retained by disabling the default filter on negative smoothing lengths:

```julia
data_list = read_phantom("dumpfile_00000"; ignore_inactive = false)
```

Global header quantities from the dump file are stored in `data.params`, while particle columns are stored in `data.dfdata` and exposed through the `ParticleDataFrame` indexing interface.



## ParticleDataFrame

`ParticleDataFrame` is a light wrapper around `DataFrames.DataFrame` together with a parameter dictionary:

```julia
struct ParticleDataFrame
    dfdata :: DataFrame
    params :: Dict
end
```

Column access follows the same conventions as `DataFrame`:

```julia
x = gas[!, :x]          # direct reference to a column
h = gas[:, :h]          # copied column access
subset = gas[gas[!, :h] .> 0, :]
cols = names(gas)
```

The package also provides a set of particle-side convenience utilities, including:

- `print_params`
- `get_dim`
- `get_time`
- `get_code_unit`
- `get_npart`
- `COM2star!`
- `star2COM!`
- `set_zaxis_orientation!`
- `add_rho!`
- `add_cylindrical!`
- `add_kinetic_energy!`
- `add_potential_energy!`
- `add_bounded_flag!`

For example, to move a particle dataset into the frame centred on a sink particle:

```julia
data_list = read_phantom("dumpfile_00000")
COM2star!(data_list, 1)
```



## Building Partia inputs

`ParticleIO.jl` also provides the `PartiaAdapter` layer, which constructs a `Partia` `InterpolationInput` and its corresponding `InterpolationCatalog` directly from a `ParticleDataFrame`.

There are two supported mass sources:

- `MassFromParams(:mass)` uses a constant particle mass stored in `data.params`
- `MassFromColumn(:m)` uses a particle-mass column stored in the table

For a dataset read directly from `read_phantom`, the usual path is to use the mass stored in the global parameter dictionary:

```julia
using ParticleIO

data = read_phantom("dumpfile_00000", "all")[1]

input, catalog = build_input(
    data,
    MassFromParams(:mass);
    scalars = (:u, :vx, :vy, :vz),
    gradients = (:rho,),
    divergences = (:v,),
    curls = (:v,),
)
```

If the particle mass is instead stored explicitly in a particle column, use:

```julia
input, catalog = build_input(
    data,
    MassFromColumn(:m);
    scalars = (:u,),
)
```

In both cases, the adapter reads the base interpolation fields from the standard particle columns:

- `:x`, `:y`, `:z`
- `:h`
- `:rho`

and then materializes any extra scalar or vector columns required by the requested `scalars`, `gradients`, `divergences`, and `curls`.

This is the intended bridge from particle-side tabular data in `ParticleIO.jl` to grid or point-sampling workflows in `Partia.jl`.
