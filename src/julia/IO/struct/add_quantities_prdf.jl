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
end

## Add extra quantities
@inline function _cylindrical(x :: V, y :: V, vx :: V, vy :: V) where {TF <: AbstractFloat, V <: AbstractVector{TF}}
    s  = similar(x)    
    ϕ  = similar(x)   
    vs = similar(x)   
    vϕ = similar(x) 
    @inbounds @simd for i in eachindex(s, ϕ, vs, vϕ)
        xi = x[i]; yi = y[i]; vxi = vx[i]; vyi = vy[i]
        si, ϕi = _cart2cylin(xi, yi)
        vsi, vϕi = _vector_cart2cylin(ϕi, vxi, vyi)

        s[i]  = si
        ϕ[i]  = ϕi
        vs[i] = vsi
        vϕ[i] = vϕi
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
end

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
end

"""
    add_rho!(data::PhantomRevealerDataFrame)
Add the local density of disk for each particles

# Parameters
- `data :: PhantomRevealerDataFrame`: The SPH data that is stored in `PhantomRevealerDataFrame` 
"""
function add_rho!(data::PhantomRevealerDataFrame)
    particle_mass = data.params[:mass]
    hfact = data.params[:hfact]
    d = get_dim(data)
    data.dfdata[!, :rho] = particle_mass .* (hfact ./ data.dfdata[!, :h]) .^ (d)
end


function _Kepelarian_azimuthal_velocity!(s :: V, vϕ :: V, μ :: TF) where {TF <: AbstractFloat, V <: AbstractVector{TF}}
    vϕk = similar(s)
    vrelϕ = similar(s)
    sqrtμ = sqrt(μ)
    @inbounds @simd for i in eachindex(s, vϕ)
        si = s[i]; vϕi = vϕ[i]
        vϕki = sqrtμ * sqrt(inv(si))
        vrelϕ[i] = vϕi - vϕki
        vϕk[i] = vϕki
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
    if !(hasproperty(data.dfdata, :s))
        add_cylindrical!(data)
    end
    G = get_unit_G(data)
    M = data.params[:Origin_sink_mass][]
    μ = G * M
    s = data.dfdata.s
    vϕ = data.dfdata.vϕ
    vϕk, vrelϕ = _Kepelarian_azimuthal_velocity!(s, vϕ, μ)
    data.dfdata.vϕk = vϕk
    data.dfdata.vrelϕ = vrelϕ
end