# Distorted Dusty Disc Analysis Toolkits / PhantomRevealer.jl

> Status: This repository is no longer the main development repository.  
> The original PhantomRevealer.jl / Distorted Dusty Disc Analysis Toolkits codebase has been reorganized and moved to the [AstroPostprocess](https://github.com/AstroPostprocess) GitHub organization.

Distorted Dusty Disc Analysis Toolkits, formerly known as `PhantomRevealer.jl`, was a Julia monorepo for SPH-based dusty-disc analysis, originating from the legacy package PhantomRevealer.jl (Su et al. 2026).

The code has since been split into several separate packages under the AstroPostprocess organization:

- [Partia.jl](https://github.com/AstroPostprocess/Partia.jl): Core LBVH-based SPH interpolation and grid-analysis tools, including kernels, neighbour search, interpolation, and structured outputs.
- [ParticleIO.jl](https://github.com/AstroPostprocess/ParticleIO.jl): Particle-data front-end companion to Partia.jl, including Phantom data handling, particle containers, and adapters into Partia.jl.
- [StreamingInstability.jl](https://github.com/AstroPostprocess/StreamingInstability.jl): Classical linear streaming-instability growth-rate analysis tools for dust-gas mixtures in protoplanetary discs.
- [TinyEigvals.jl](https://github.com/AstroPostprocess/TinyEigvals.jl): A small-matrix eigensolver package used by StreamingInstability.jl. This package is already available from the Julia General registry.

SpiralDetection.jl has not been reimplemented yet. The previous working API can still be found on the OptionalGLMakie branch of this repository if needed.

## Recommended citation

If this toolkit or its successor packages are useful for your work, please cite:

Su et al. 2026, MNRAS, 547, 2, stag173.
