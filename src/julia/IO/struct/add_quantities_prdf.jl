"""
Modifing PhantomRevealerDataFrame for several purposes
    by Wei-Shan Su
    September 28, 2025

Those methods with prefix `add` would store the result into the original data, and prefix `get` would return the value. 
Becarful, the methods with suffix `!` would change the inner state of its first argument!
"""


# Method
## Distance measurements
"""
    get_rnorm_ref(data::PhantomRevealerDataFrame, reference_position::NTuple{3, Float64})
Get the array of distance between particles and the reference_position.

# Parameters
- `data :: PhantomRevealerDataFrame`: The SPH data that is stored in `PhantomRevealerDataFrame` 
- `reference_position::NTuple{3, Float64}`: The reference point to estimate the distance.

# Returns
- `Vector`: The array of distance between particles and the reference_position.
"""
function get_rnorm_ref(data::PhantomRevealerDataFrame, reference_position::NTuple{3, TF}) where {TF <: AbstractFloat}
    x = data.dfdata.x; y = data.dfdata.y; z = data.dfdata.z
    rnorm :: Vector{TF} = Euclidean_distance(x, y, z, reference_position)
    return rnorm
end

"""
    get_snorm_ref(data::PhantomRevealerDataFrame, reference_position::NTuple{2, TF})
Get the array of distance between particles and the reference_position ON THE XY-PLANE PROJECTION.

# Parameters
- `data :: PhantomRevealerDataFrame`: The SPH data that is stored in `PhantomRevealerDataFrame` 
- `reference_position::NTuple{2, TF}`: The reference point to estimate the distance.

# Returns
- `Vector`: The array of distance between particles and the reference_position ON THE XY-PLANE PROJECTION.
"""
function get_snorm_ref(data::PhantomRevealerDataFrame, reference_position::NTuple{2, TF}) where {TF <: AbstractFloat}
    x = data.dfdata.x; y = data.dfdata.y
    rnorm :: Vector{TF} = Euclidean_distance(x, y, reference_position)
    return rnorm
end

"""
    get_rnorm(data::PhantomRevealerDataFrame)
Get the array of distance between particles and the origin ON THE XY-PLANE PROJECTION.

# Parameters
- `data :: PhantomRevealerDataFrame`: The SPH data that is stored in `PhantomRevealerDataFrame` 

# Returns
- `Vector`: The array of distance between particles and the origin ON THE XY-PLANE PROJECTION.
"""
function get_rnorm(data::PhantomRevealerDataFrame)
    return get_rnorm_ref(data, (0.0, 0.0, 0.0))
end

"""
    get_snorm(data::PhantomRevealerDataFrame)
Get the array of distance between particles and the origin ON THE XY-PLANE PROJECTION.

# Parameters
- `data :: PhantomRevealerDataFrame`: The SPH data that is stored in `PhantomRevealerDataFrame` 

# Returns
- `Vector`: The array of distance between particles and the origin ON THE XY-PLANE PROJECTION.
"""
function get_snorm(data::PhantomRevealerDataFrame)
    return get_snorm_ref(data, (0.0, 0.0))
end

## Shift coordinate
### Translation
function _apply_coordinate_shift!(data :: PhantomRevealerDataFrame, new_origin :: NTuple{6, TF}) where {TF <: AbstractFloat}
    xo, yo, zo, vxo, vyo, vzo = new_origin
    x  = data.dfdata.x
    y  = data.dfdata.y
    z  = data.dfdata.z
    vx = data.dfdata.vx
    vy = data.dfdata.vy
    vz = data.dfdata.vz
    @inbounds @simd for i in eachindex(x)
        xi = x[i]; yi = y[i]; zi = z[i]
        vxi = vx[i]; vyi = vy[i]; vzi = vz[i]

        x[i] = xi - xo
        y[i] = yi - yo
        z[i] = zi - zo
        vx[i] = vxi - vxo
        vy[i] = vyi - vyo
        vz[i] = vzi - vzo
    end
    return nothing
end

"""
    COM2star!(data_list :: V,  sink_particle_id :: Int) where {D <: PhantomRevealerDataFrame, V <: AbstractVector{D}}
Transfer the coordinate to another coordinate with locating star at the origin.
Assuming the sink data is stored as the final `PhantomRevealerDataFrame` in the Vector

# Parameters
- `data_list :: V`: The array which contains all of the data that would be transfered
- `sink_particle_id :: Int`: The id of star that would be located at the origin.

# Example
```julia
# Transfer to the primary star-based coodinate(id=1)
prdf_list = read_phantom(dumpfile_00000)
COM2star!(prdf_list, 1)
```
"""
function COM2star!(data_list :: V, sink_particle_id :: Int) where {D <: PhantomRevealerDataFrame, V <: AbstractVector{D}}
    sinks_data = data_list[end]
    new_origin = (sinks_data.dfdata.x[sink_particle_id], 
                  sinks_data.dfdata.y[sink_particle_id],
                  sinks_data.dfdata.z[sink_particle_id],
                  sinks_data.dfdata.vx[sink_particle_id],
                  sinks_data.dfdata.vy[sink_particle_id],
                  sinks_data.dfdata.vz[sink_particle_id])

    @inbounds for data in data_list
        _apply_coordinate_shift!(data, new_origin)
    end
    sinks_data.params[:COM_coordinate] .-= new_origin
    sinks_data.params[:Origin_sink_id][] = sink_particle_id
    sinks_data.params[:Origin_sink_mass][] = sinks_data.dfdata.m[sink_particle_id]
    return nothing
end

"""
    star2COM!(data_list :: V) where {D <: PhantomRevealerDataFrame, V <: AbstractVector{D}}
Transfer the coordinate to COM coordinate.

# Parameters
- `data_list :: V`: The array which contains all of the data that would be transfered

# Example
```julia
# Transfer to the primary star-based coodinate(id=1), and then transfer back.
prdf_list = read_phantom(dumpfile_00000)
println(prdf_list[1].params[:Origin_sink_id][])  # print: -1
COM2star!(prdf_list, 1)
println(prdf_list[1].params[:Origin_sink_id][])  # print: 1
star2COM!(prdf_list)
println(prdf_list[1].params[:Origin_sink_id][])  # print: -1
```
"""

function star2COM!(data_list :: V) where {D <: PhantomRevealerDataFrame, V <: AbstractVector{D}}
    COM_position = data_list[1].params[:COM_coordinate]
    new_origin = ntuple(i -> COM_position[i], 6)

    @inbounds for data in data_list
        _apply_coordinate_shift!(data, new_origin)
    end
    data_list[1].params[:COM_coordinate] .-= new_origin
    data_list[1].params[:Origin_sink_id][] = -1
    data_list[1].params[:Origin_sink_mass][] = NaN64
    return nothing
end

### Rotation
@inline function _rotational_matrix(lx :: TF, ly :: TF, lz :: TF) :: NTuple{9, TF} where {TF <: AbstractFloat}
    # Normalize target axis
    lnorm = sqrt(lx^2 + ly^2 + lz^2)
    lx /= lnorm
    ly /= lnorm
    lz /= lnorm

    # Constructing rotating elements
    ρ = sqrt(lx^2 + lz^2)
    if ρ ≤ eps(TF)
        cosϕy = one(TF) ;  sinϕy = zero(TF)
        cosϕx = zero(TF);  sinϕx = sign(ly)
        cosψ  = one(TF) ;  sinψ = zero(TF)
    else
        invρ  = 1/ρ
        cosϕy = lz*invρ
        sinϕy = -lx*invρ   
        cosϕx = ρ
        sinϕx = ly
        ψ = atan(lx, -ly*lz)
        cosψ, sinψ = cos(ψ), sin(ψ)
    end

    # Rotational matrix R
    r11 = cosψ * cosϕy + sinψ * sinϕx * sinϕy
    r12 = sinψ * cosϕx
    r13 = cosψ * sinϕy - sinψ * sinϕx * cosϕy

    r21 = -sinψ * cosϕy + cosψ * sinϕx * sinϕy
    r22 =  cosψ * cosϕx
    r23 = -sinψ * sinϕy - cosψ * sinϕx * cosϕy

    r31 = -cosϕx * sinϕy
    r32 =  sinϕx
    r33 =  cosϕx * cosϕy
    return (r11,r12,r13,r21,r22,r23,r31,r32,r33)
end 

function _apply_zaxis_orientation!(data :: PhantomRevealerDataFrame, R :: NTuple{9, TF}) where {TF <: AbstractFloat}
    # Inplace rotation
    x  = data.dfdata.x
    y  = data.dfdata.y
    z  = data.dfdata.z
    vx = data.dfdata.vx
    vy = data.dfdata.vy
    vz = data.dfdata.vz
    r11,r12,r13,r21,r22,r23,r31,r32,r33 = R
    @inbounds @simd for i in eachindex(x)
        xi,yi,zi = x[i],y[i],z[i]
        vxi,vyi,vzi = vx[i],vy[i],vz[i]
        x[i]  = muladd(r13,zi, muladd(r12,yi, r11*xi))
        y[i]  = muladd(r23,zi, muladd(r22,yi, r21*xi))
        z[i]  = muladd(r33,zi, muladd(r32,yi, r31*xi))
        vx[i] = muladd(r13,vzi, muladd(r12,vyi, r11*vxi))
        vy[i] = muladd(r23,vzi, muladd(r22,vyi, r21*vxi))
        vz[i] = muladd(r33,vzi, muladd(r32,vyi, r31*vxi))
    end
    return nothing
end

"""
    set_zaxis_orientation!(data::PhantomRevealerDataFrame, target_zaxis::NTuple{3,TF}; inverse::Bool=false) where {TF<:AbstractFloat}

Rotate all positions (x,y,z) and velocities (vx,vy,vz) in-place so that the z-axis aligns with the given `target_zaxis`. The rotation is orthogonal (det=1), preserving lengths and inner products. By construction, the new x-axis lies in the original xy-plane:
    x' = normalize(ẑ × l̂),   y' = z' × x',   z' = l̂

where ẑ = (0,0,1) and l̂ = normalized target_zaxis.

        1      0      0
Rx = [  0   cos(ϕx) -sin(ϕx)]
        0   sin(ϕx)  cos(ϕx) 

    cos(ϕy)  0     sin(ϕy)
Ry = [  0      1      0     ]
    -sin(ϕy)  0     cos(ϕy)

# Parameters
- `data::PhantomRevealerDataFrame`: SPH data container with fields `x,y,z,vx,vy,vz`.
- `target_zaxis::NTuple{3,TF}`: Vector specifying the new z-axis (not required to be normalized).

# Keyword Arguments
- `inverse::Bool=false`: If `false`, apply the forward rotation (v' = R * v).
                         If `true`, apply the inverse rotation (v' = R' * v).

"""
function set_zaxis_orientation!(data :: PhantomRevealerDataFrame, target_zaxis :: NTuple{3, TF}; inverse :: Bool = false) where {TF<: AbstractFloat}
    Rdefault = _rotational_matrix(target_zaxis...)
    if inverse
        R = (Rdefault[1], Rdefault[4], Rdefault[7], Rdefault[2], Rdefault[5], Rdefault[8], Rdefault[3], Rdefault[6], Rdefault[9])
    else
        R = Rdefault
    end    
    _apply_zaxis_orientation!(data, R)
    return nothing
end

"""
    set_zaxis_orientation!(data_list :: V, target_zaxis :: NTuple{3, TF}; inverse :: Bool = false) where {TF<: AbstractFloat, D <: PhantomRevealerDataFrame, V <: AbstractVector{D}}

Rotate all positions (x,y,z) and velocities (vx,vy,vz) in-place so that the z-axis aligns with the given `target_zaxis`. The rotation is orthogonal (det=1), preserving lengths and inner products. By construction, the new x-axis lies in the original xy-plane:
    x' = normalize(ẑ × l̂),   y' = z' × x',   z' = l̂

where ẑ = (0,0,1) and l̂ = normalized target_zaxis.

# Parameters
- `data_list :: V`: The array which contains all of the data that would be transfered
- `target_zaxis::NTuple{3,TF}`: Vector specifying the new z-axis (not required to be normalized).

# Keyword Arguments
- `inverse::Bool=false`: If `false`, apply the forward rotation (v' = R * v).
                         If `true`, apply the inverse rotation (v' = R' * v).

"""
function set_zaxis_orientation!(data_list :: V, target_zaxis :: NTuple{3, TF}; inverse :: Bool = false) where {TF<: AbstractFloat, D <: PhantomRevealerDataFrame, V <: AbstractVector{D}}
    Rdefault = _rotational_matrix(target_zaxis...)
    if inverse
        R = (Rdefault[1], Rdefault[4], Rdefault[7], Rdefault[2], Rdefault[5], Rdefault[8], Rdefault[3], Rdefault[6], Rdefault[9])
    else
        R = Rdefault
    end
    @inbounds for data in data_list
        _apply_zaxis_orientation!(data, R)
    end
    return nothing
end

## Add extra quantities
### Density
@inline function _density(h :: V, m :: TF, hfact :: TF, ::Val{2}) where {TF <: AbstractFloat, V <: AbstractVector}
    """ Valid only if mb = m (mb is a constant)"""
    ρ = similar(h)
    mhfactd = m * hfact * hfact
    @inbounds @simd for i in eachindex(ρ)
        hi = h[i]
        invhi = inv(hi)
        invhid = invhi * invhi
        ρi = mhfactd * invhid
        ρ[i] = ρi
    end
    return ρ
end

@inline function _density(h :: V, m :: TF, hfact :: TF, ::Val{3}) where {TF <: AbstractFloat, V <: AbstractVector}
    """ Valid only if mb = m (mb is a constant)"""
    ρ = similar(h)
    mhfactd = m * hfact * hfact * hfact
    @inbounds @simd for i in eachindex(ρ)
        hi = h[i]
        invhi = inv(hi)
        invhid = invhi * invhi * invhi
        ρi = mhfactd * invhid
        ρ[i] = ρi
    end
    return ρ
end

"""
    add_rho!(data::PhantomRevealerDataFrame)
Add the local density of disk for each particles.

**Note**: This function is invalid when per-particle masses are used.

# Parameters
- `data :: PhantomRevealerDataFrame`: The SPH data that is stored in `PhantomRevealerDataFrame` 
"""
function add_rho!(data::PhantomRevealerDataFrame)
    if (hasproperty(data.dfdata, "m"))
        @warn("This function assumes all particles share the same mass. It is invalid if per-particle masses are used.")
    end
    if !(haskey(data.params, :mass))
        error("KeyError: Missing required parameter :mass in data.params.")
    end
    
    particle_mass = data.params[:mass]
    hfact = data.params[:hfact]
    d = get_dim(data)
    TF = typeof(particle_mass)
    h = TF.(data.dfdata.h)
    ρ = _density(h, m, hfact, Val(d))
    data.dfdata.rho = ρ
    return nothing
end

### Norm of coordinate and velocity w.r.t. origin
"""
    add_norm!(data::PhantomRevealerDataFrame)
Add the length of position vector and velocity vector in 3D.

# Parameters
- `data :: PhantomRevealerDataFrame`: The SPH data that is stored in `PhantomRevealerDataFrame` 
"""
function add_norm!(data::PhantomRevealerDataFrame)
    x = data.dfdata.x; y = data.dfdata.y; z = data.dfdata.z
    vx = data.dfdata.vx; vy = data.dfdata.vy; vz = data.dfdata.vz

    r = Euclidean_distance(x, y, z, (0.0,0.0,0.0))
    vr = Euclidean_distance(vx, vy, vz, (0.0,0.0,0.0))

    data.dfdata.r = r
    data.dfdata.vr = vr
    return nothing
end

### Energy
function _kinetic_energy(vx :: V, vy :: V, vz :: V, m :: TF) where {TF <: AbstractFloat, V <: AbstractVector{TF}}
    KE = similar(vx)
    ml2 = TF(0.5) * m
    @inbounds @simd for i in eachindex(KE)
        vxi = vx[i]; vyi = vy[i]; vzi = vz[i];
        vi2 = vxi * vxi + vyi * vyi + vzi * vzi

        KEi = ml2 * vi2
        KE[i] = KEi
    end
    return KE
end

function _kinetic_energy(vx :: V, vy :: V, vz :: V, m :: V) where {TF <: AbstractFloat, V <: AbstractVector{TF}}
    KE = similar(vx)
    C = TF(0.5)
    @inbounds @simd for i in eachindex(KE)
        vxi = vx[i]; vyi = vy[i]; vzi = vz[i];
        mi = m[i]
        vi2 = vxi * vxi + vyi * vyi + vzi * vzi

        KEi = C * mi * vi2
        KE[i] = KEi
    end
    return KE
end


"""
    add_kinetic_energy!(data::PhantomRevealerDataFrame; specific_mass_column::Symbol = :m)

Compute and add the kinetic energy of each particle in `data` for the current frame.

# Parameters
- `data :: PhantomRevealerDataFrame`: SPH particle data stored in `PhantomRevealerDataFrame`.

# Keyword Arguments
| Name | Default | Description |
|------|----------|-------------|
| `specific_mass_column` | `:m` | Symbol of the column storing per-particle masses. If missing, a global mass in `data.params[:mass]` will be used instead. |
"""
function add_kinetic_energy!(data::PhantomRevealerDataFrame; specific_mass_column :: Symbol = :m)
    vx = data.dfdata.vx
    vy = data.dfdata.vy
    vz = data.dfdata.vz
    TF = eltype(vx)

    if !(hasproperty(data.dfdata, specific_mass_column))
        if (haskey(data.params, :mass))
            mass = TF(data.params[:mass])
            KE = _kinetic_energy(vx, vy, vz, mass)
            data.dfdata.KE = KE
        else
            error("ArgumentError: Mass is missing from this data")
        end
    else
        masses = data.dfdata[!, specific_mass_column]
        KE = _kinetic_energy(vx, vy, vz, masses)
        data.dfdata.KE = KE
    end
    return nothing
end

@inline function _potential_energy(Δx :: TF, Δy :: TF, Δz :: TF, m :: TF, μ :: TF) where {TF <: AbstractFloat}
    C = - μ * m
    Δr = sqrt(Δx * Δx + Δy * Δy + Δz * Δz)
    invΔr = inv(Δr)
    return C * invΔr
end

function _potential_energy(x :: V, y :: V, z :: V, m :: TF, xt :: TF, yt :: TF, zt :: TF, μ :: TF) where {TF <: AbstractFloat, V <: AbstractVector{TF}}
    PE = similar(x)
    @inbounds @simd for i in eachindex(PE)
        xi = x[i]; yi = y[i]; zi = z[i];
        Δxi = xi - xt
        Δyi = yi - yt
        Δzi = zi - zt
        PEi = _potential_energy(Δxi, Δyi, Δzi, m, μ)
        PE[i] = PEi
    end
    return PE
end

function _potential_energy(x :: V, y :: V, z :: V, m :: V, xt :: TF, yt :: TF, zt :: TF, μ :: TF) where {TF <: AbstractFloat, V <: AbstractVector{TF}}
    PE = similar(x)
    @inbounds @simd for i in eachindex(PE)
        xi = x[i]; yi = y[i]; zi = z[i];
        mi = m[i]
        Δxi = xi - xt
        Δyi = yi - yt
        Δzi = zi - zt
        PEi = _potential_energy(Δxi, Δyi, Δzi, mi , μ)
        PE[i] = PEi
    end
    return PE
end

"""
    add_potential_energy!(data::PhantomRevealerDataFrame, sink_data::PhantomRevealerDataFrame; pecific_mass_column::Symbol = :m, store_sinks :: V = Int[]) where {V <: AbstractVector{<:Integer}}

Compute and add the gravitational potential energy of all gas particles in `data` with respect to each sink particle in `sink_data`.

# Parameters
- `data :: PhantomRevealerDataFrame`: SPH particle data stored in `PhantomRevealerDataFrame`.
- `sink_data :: PhantomRevealerDataFrame`: Sink particle data used as potential sources.

# Keyword Arguments
| Name | Default | Description |
|------|----------|-------------|
| `specific_mass_column` | `:m` | Symbol of the column storing per-particle masses. If missing, a global mass in `data.params[:mass]` will be used instead. |
| `store_sinks` | `Int[]` | Indices of sink particles for which individual potential energy columns (`PEₙ`) will be stored. If empty, only the total potential energy (`PEtot`) is added. |


"""
function add_potential_energy!(data::PhantomRevealerDataFrame, sink_data :: PhantomRevealerDataFrame; specific_mass_column::Symbol = :m, store_sinks :: V = Int[]) where {V <: AbstractVector{<:Integer}}
    x = data.dfdata.x
    y = data.dfdata.y
    z = data.dfdata.z

    xt = sink_data.dfdata.x
    yt = sink_data.dfdata.y
    zt = sink_data.dfdata.z
    mt = sink_data.dfdata.m

    TF = eltype(x)

    G = get_unit_G(data)
    num_part = get_npart(data)
    num_sink = get_npart(sink_data)
    
    use_threads = (nthreads() > 1) && (nthreads() ÷ 2 > num_sink)

    if !(hasproperty(data.dfdata, specific_mass_column))
        if (haskey(data.params, :mass))
            mass = TF(data.params[:mass])
            if use_threads
                PEs = Vector{Vector{TF}}(undef, num_sink)
                @threads for n in 1:num_sink
                    PEs[n] = _potential_energy(x, y, z, mass, xt[n], yt[n], zt[n], G * mt[n])
                end
            else
                PEs = [_potential_energy(x, y, z, mass, xt[n], yt[n], zt[n], G * mt[n]) for n in 1:num_sink]
            end
            PEtot = zeros(TF, num_part)
            for n in 1:num_sink
                @inbounds @simd for i in 1:num_part
                    PEtot[i] += PEs[n][i]
                end
            end
            data.dfdata.PEtot = PEtot
            for n in 1:num_sink
                if n in store_sinks
                    data.dfdata[!, "PE$(n)"] = PEs[n]
                end
            end
        else
            error("ArgumentError: Mass is missing from this data")
        end
    else
        masses = data.dfdata[!, specific_mass_column]
        if use_threads
                PEs = Vector{Vector{TF}}(undef, num_sink)
                @threads for n in 1:num_sink
                    PEs[n] = _potential_energy(x, y, z, masses, xt[n], yt[n], zt[n], G * mt[n])
                end
            else
                PEs = [_potential_energy(x, y, z, masses, xt[n], yt[n], zt[n], G * mt[n]) for n in 1:num_sink]
            end
        PEtot = zeros(TF, num_part)
        for n in 1:num_sink
            @inbounds @simd for i in 1:num_part
                PEtot[i] += PEs[n][i]
            end
        end
        data.dfdata.PEtot = PEtot
        for n in 1:num_sink
            if n in store_sinks
                data.dfdata[!, "PE$(n)"] = PEs[n]
            end
        end
    end
    return nothing
end

function _bounded_flag(KE::V, PEn::V) where {TF <: AbstractFloat, V <: AbstractVector{TF}}
    bflag = zeros(Bool, length(KE))
    @inbounds @simd for i in eachindex(KE)
        Eni = KE[i] + PEn[i]
        bflag[i] = (Eni < 0)
    end
    return bflag
end

"""
    add_bounded_flag!(data::PhantomRevealerDataFrame; check_sinks::V = [1]) where {V <: AbstractVector{<:Integer}}

Compute and add boundedness flags for each particle relative to one or more sink particles.

A particle is considered *bound* to a sink if its total energy (kinetic + potential) relative to that sink is negative,  
i.e. `E_total = KE + PEₙ < 0`.

# Parameters
- `data :: PhantomRevealerDataFrame`:  
  SPH particle data stored in a `PhantomRevealerDataFrame`. Must contain kinetic energy `KE` and potential energy columns `PEₙ`.

# Keyword Arguments
| Name | Default | Description |
|------|----------|-------------|
| `check_sinks` | `[1]` | Vector of sink indices (e.g. `[1,2,3]`) for which boundedness flags will be computed. Each will generate a `bflagₙ` column. |

"""
function add_bounded_flag!(data::PhantomRevealerDataFrame; check_sinks :: V = Int[1]) where {V <: AbstractVector{<:Integer}}
    if !hasproperty(data.dfdata, :KE)
        error("ArgumentError: Kinetic Energy of particles is missing!")
    end
    for n in check_sinks
        if !hasproperty(data.dfdata, "PE$(n)")
            error("ArgumentError: Potential Energy to sink $(n) is missing!")
        end
    end

    KE = data.dfdata.KE
    PEs = [data.dfdata[!, "PE$(n)"] for n in check_sinks]
    flags = [_bounded_flag(KE, PEs[k]) for k in eachindex(check_sinks)]

    for (k,n) in enumerate(check_sinks)
        data.dfdata[!, "bflag$(n)"] = flags[k]
    end
    return nothing
end


### Cylindrical coordinate
function _cylindrical(x :: V, y :: V, vx :: V, vy :: V) where {TF <: AbstractFloat, V <: AbstractVector{TF}}
    s  = similar(x)    
    ϕ  = similar(x)   
    vs = similar(x)   
    vϕ = similar(x) 
    @inbounds @simd for i in eachindex(s, ϕ, vs, vϕ)
        xi = x[i]; yi = y[i]; vxi = vx[i]; vyi = vy[i]
        si, ϕi = _cart2cylin(xi, yi)
        vsi, vϕi = _vector_cart2cylin(ϕi, vxi, vyi)

        @inbounds begin
            s[i]  = si; ϕ[i]  = ϕi; vs[i] = vsi; vϕ[i] = vϕi
        end
    end
    return s, ϕ, vs, vϕ
end

"""
    add_cylindrical!(data::PhantomRevealerDataFrame)
Add the cylindrical/polar coordinate (s,ϕ) and corresponding velocity (vs, vϕ) into the data

# Parameters
- `data :: PhantomRevealerDataFrame`: The SPH data that is stored in `PhantomRevealerDataFrame` 
"""
function add_cylindrical!(data::PhantomRevealerDataFrame)
    x  = data.dfdata.x 
    y  = data.dfdata.y 
    vx = data.dfdata.vx
    vy = data.dfdata.vy

    s, ϕ, vs, vϕ = _cylindrical(x, y, vx, vy)  

    data.dfdata.s  = s
    data.dfdata.ϕ  = ϕ
    data.dfdata.vs = vs
    data.dfdata.vϕ = vϕ
    return nothing
end

### Kepelarian azimuthal velocity (vϕk) and the relative azimuthal velocity (vϕ - vϕk)
function _Kepelarian_azimuthal_velocity(s :: V, vϕ :: V, μ :: TF) where {TF <: AbstractFloat, V <: AbstractVector{TF}}
    vϕk = similar(s)
    vrelϕ = similar(s)
    sqrtμ = sqrt(μ)
    @inbounds @simd for i in eachindex(vϕk, vrelϕ)
        si = s[i]; vϕi = vϕ[i]
        vϕki = sqrtμ * sqrt(inv(si))
        vrelϕi = vϕi - vϕki
        @inbounds begin
            vrelϕ[i] = vrelϕi
            vϕk[i] = vϕki
        end
        
    end
    return vϕk, vrelϕ
end
"""
    add_Kepelarian_azimuthal_velocity!(data::PhantomRevealerDataFrame)
Add the Kepelarian azimuthal velocity for each particles.

# Parameters
- `data :: PhantomRevealerDataFrame`: The SPH data that is stored in `PhantomRevealerDataFrame`
"""
function add_Kepelarian_azimuthal_velocity!(data::PhantomRevealerDataFrame)
    if !(haskey(data.params, :Origin_sink_id)) || (data.params[:Origin_sink_id] == -1)
        error(
            "OriginLocatedError: Wrong origin located. Please use COM2star!() to transfer the coordinate.",
        )
    end
    if !(hasproperty(data.dfdata, :s))
        add_cylindrical!(data)
    end
    G = get_unit_G(data)
    M1 = data.params[:Origin_sink_mass][]
    μ = G * M1
    s = data.dfdata.s
    vϕ = data.dfdata.vϕ
    vϕk, vrelϕ = _Kepelarian_azimuthal_velocity!(s, vϕ, μ)
    data.dfdata.vϕk = vϕk
    data.dfdata.vrelϕ = vrelϕ
    return nothing
end

### Kepelarian angular velocity (Ωk)
function _Kepelarian_angular_velocity(x :: V, y :: V, z :: V, μ :: TF) where {TF <: AbstractFloat, V <: AbstractVector{TF}}
    Ωk = similar(x)
    sqrtμ = sqrt(μ)
    @inbounds @simd for i in eachindex(Ωk)
        xi = x[i]; yi = y[i]; zi = z[i]
        ri = sqrt(xi * xi + yi * yi + zi * zi)
        invri3 = inv(ri * ri * ri)
        Ωki = sqrtμ * sqrt(invri3)
        Ωk[i] = Ωki
    end
    return Ωk
end

"""
    add_Kepelarian_angular_velocity!(data::PhantomRevealerDataFrame)
Add the Kepelarian angular velocity for each particles.

# Parameters
- `data :: PhantomRevealerDataFrame`: The SPH data that is stored in `PhantomRevealerDataFrame`
"""
function add_Kepelarian_angular_velocity!(data::PhantomRevealerDataFrame)
    if !(haskey(data.params, :Origin_sink_id)) || (data.params[:Origin_sink_id] == -1)
        error(
            "OriginLocatedError: Wrong origin located. Please use COM2star!() to transfer the coordinate.",
        )
    end
    G = get_unit_G(data)
    M1 = data.params[:Origin_sink_mass][]
    μ = G * M1
    x = data.dfdata.x
    y = data.dfdata.y
    z = data.dfdata.z
    Ωk = _Kepelarian_angular_velocity(x, y, z, μ)
    data.dfdata.Ωk = Ωk
    return nothing
end

### Eccentricity
function _eccentricity(x :: V, y :: V, z :: V, vx :: V, vy :: V, vz :: V, μ :: TF) where {TF <: AbstractFloat, V <: AbstractVector{TF}}
    e  = similar(x)     
    invµ = inv(μ)
    @inbounds @simd for i in eachindex(e)
        xi = x[i]; yi = y[i]; zi=z[i]; vxi = vx[i]; vyi = vy[i]; vzi = vz[i]
        
        ri = sqrt(xi * xi + yi * yi + zi * zi)
        vri2 = vxi * vxi + vyi * vyi + vzi * vzi

        ridotvi = xi * vxi + yi * vyi + zi * vzi
        
        invri = inv(ri)

        ridotvilµ = ridotvi * invµ
        vri2lµminvri = vri2 * invµ - invri
        

        exi = xi * vri2lµminvri - vxi * ridotvilµ
        eyi = yi * vri2lµminvri - vyi * ridotvilµ
        ezi = zi * vri2lµminvri - vzi * ridotvilµ

        e[i] = sqrt(exi * exi + eyi * eyi + ezi * ezi)
    end
    return e
end

"""
    add_eccentricity!(data::PhantomRevealerDataFrame)
Add the eccentricity for each particle with respect to current origin.

# Parameters
- `data :: PhantomRevealerDataFrame`: The SPH data that is stored in `PhantomRevealerDataFrame`
"""
function add_eccentricity!(data::PhantomRevealerDataFrame)
    if !(haskey(data.params, :Origin_sink_id)) || (data.params[:Origin_sink_id] == -1)
        error(
            "OriginLocatedError: Wrong origin located. Please use COM2star!() to transfer the coordinate.",
        )
    end
    G = get_unit_G(data)
    M1 = data.params[:Origin_sink_mass][]
    μ = G * M1
    x = data.dfdata.x
    y = data.dfdata.y
    z = data.dfdata.z
    vx = data.dfdata.vx
    vy = data.dfdata.vy
    vz = data.dfdata.vz
    e = _eccentricity(x, y, z, vx, vy, vz, μ)
    data.dfdata.e = e 
    return nothing
end

# Specific angular momentum (lx, ly, lz, l)
function _specific_angular_momentum(x :: V, y :: V, z :: V, vx :: V, vy :: V, vz :: V) where {TF <: AbstractFloat, V <: AbstractVector{TF}}
    lx  = similar(x)     
    ly  = similar(x)     
    lz  = similar(x) 
    l   = similar(x)    

    @inbounds @simd for i in eachindex(lx, ly, lz, l)
        xi = x[i]; yi = y[i]; zi=z[i]; vxi = vx[i]; vyi = vy[i]; vzi = vz[i]
        
        lxi = yi * vzi - zi * vyi
        lyi = zi * vxi - xi * vzi
        lzi = xi * vyi - yi * vxi
        li  = sqrt(lxi * lxi + lyi * lyi + lzi * lzi)

        @inbounds begin
            lx[i] = lxi; ly[i] = lyi; lz[i] = lzi; l[i] = li
        end
    end
    return lx, ly, lz, l
end

"""
    add_specific_angular_momentum!(data::PhantomRevealerDataFrame)

Compute and add the specific angular momentum vector **l = r × v** for each particle in the current frame.

# Parameters
- `data :: PhantomRevealerDataFrame`: The SPH data that is stored in `PhantomRevealerDataFrame`
"""
function add_specific_angular_momentum!(data::PhantomRevealerDataFrame)
    x = data.dfdata.x
    y = data.dfdata.y
    z = data.dfdata.z
    vx = data.dfdata.vx
    vy = data.dfdata.vy
    vz = data.dfdata.vz
    lx, ly, lz, l = _specific_angular_momentum(x, y, z, vx, vy, vz)
    data.dfdata.lx = lx
    data.dfdata.ly = ly
    data.dfdata.lz = lz
    data.dfdata.lnorm = l
    return nothing
end

### Tilting (i)
function _tilt(x :: V, y :: V, z :: V, lx :: V, ly :: V, lz :: V, l :: V) where {TF <: AbstractFloat, V <: AbstractVector{TF}}
    tilt = similar(x)
    @inbounds @simd for i in eachindex(tilt)
        xi = x[i]  ; yi = y[i]  ; zi = z[i]
        lxi = lx[i]; lyi = ly[i]; lzi = lz[i];
        li = l[i]
        invli = inv(li)

        ri = sqrt(xi * xi + yi * yi + zi * zi)
        invri = inv(ri)
        rlproji = invli * (xi * lxi + yi * lyi + zi * lzi)              # Prevent non-normalized
        rlproji = clamp(rlproji, -1.0, 1.0)
        tilti = asin(rlproji * invri)
        tilt[i] = tilti
    end
    return tilt
end

"""
    add_tilt!(data::PhantomRevealerDataFrame)

Compute and add the inclination angle (tilt) between each particle’s position vector **r**
and its specific angular momentum vector **l**.

The tilt is defined as  
`tilt = asin((r ⋅ l) / (|r||l|))`,  
representing the local misalignment angle between a particle’s orbital plane and its angular momentum direction.

For **gravitationally bound** particles (`E < 0`), it represents the true orbital inclination of the particle’s motion.  
For **unbound** particles (`E ≥ 0`), it only indicates the instantaneous direction of angular momentum and has no stable orbital interpretation.

# Parameters
- `data :: PhantomRevealerDataFrame`:  
  The SPH particle data stored in a `PhantomRevealerDataFrame`.  
  Must contain columns `x`, `y`, `z`, and either precomputed angular momentum components `lx`, `ly`, `lz`, `lnorm`,  
  or allow them to be generated automatically by `add_specific_angular_momentum!()`.

# Behavior
- If `lx`, `ly`, or `lz` columns are missing, they are automatically computed.  
- Adds a new column `tilt` to `data.dfdata` containing the tilt angle (in radians) for each particle.
"""
function add_tilt!(data::PhantomRevealerDataFrame)
    if !(hasproperty(data.dfdata, "lx")) || !(hasproperty(data.dfdata, "ly")) || !(hasproperty(data.dfdata, "lz"))
        add_specific_angular_momentum!(data)
    end

    x = data.dfdata.x
    y = data.dfdata.y
    z = data.dfdata.z
    lx = data.dfdata.lx
    ly = data.dfdata.ly
    lz = data.dfdata.lz
    l = data.dfdata.lnorm

    tilt = _tilt(x, y, z, lx, ly, lz, l)
    data.dfdata.tilt = tilt

    return nothing
end

