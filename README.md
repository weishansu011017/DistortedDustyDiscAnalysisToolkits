# Distorted Dusty Disc Analysis Toolkits

Distorted dusty-disc analysis toolkits is a Julia monorepo for SPH-based dusty-disc analysis.

It is currently organised into the following companion packages:

- [`Partia.jl`](./Partia.jl): Core SPH interpolation and grid-analysis tools, including kernels, neighbour search, interpolation, and structured outputs.
- [`ParticleIO.jl`](./ParticleIO.jl): Particle-oriented data handling for Phantom workflows, including dump reading, particle containers, and adapters into `Partia.jl`.
- [`StreamingInstability.jl`](./StreamingInstability.jl): Linear growth-rate analysis tools for the classical streaming instability.
- [`SpiralDetection.jl`](./SpiralDetection.jl): Placeholder package for future spiral-structure detection and post-processing workflows; the implementation is not finished yet, and the previous working API can be found on the `OptionalGLMakie` branch if needed.
