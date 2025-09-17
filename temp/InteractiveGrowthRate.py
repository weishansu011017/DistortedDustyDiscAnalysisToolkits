'''
    Interactive Growth Rate estimation toolkit 
        by Wei-Shan Su Augest 8, 2024
    
    The estimation method is developed by Chen&Lin(2020) (doi=10.3847/1538-4357/ab76ca).
    Usage: python InteractiveGrowthRate.py Faceon_00XXX.h5
'''
import numpy as np
import argparse
import sys
import juliacall
from math import isclose

jl = juliacall.newmodule("PhantomRevealer")
jl.seval("using PhantomRevealer")
PhantomRevealer_path = jl.get_PhantomRevealer_path()
sys.path.insert(0, f'{PhantomRevealer_path}/src')
sys.path.insert(0, f'{PhantomRevealer_path}/src/julia')
sys.path.insert(0, f'{PhantomRevealer_path}/script')

from PhantomRevealerAnalysisResult import *
from python.pyplot_backend import *

def minimum_growth_rate(radius_au,M_solarmass,spiral_lifetime_yr=3000):
    radius_SI = radius_au*149597870700
    G_SI = 6.67e-11
    M_SI = M_solarmass*1.9891e30
    Q = 31556926
    omega = np.sqrt(G_SI*M_SI)*(radius_SI**(-3/2))
    return 1/(omega*spiral_lifetime_yr*Q)

def growth_rate_calculation(data:PhantomRevealerAnalysisResult, s_index, phi_index,msetup:dict,column_indices:dict):
    utime = data.params["utime"]
    udist = data.params["udist"]
    uv = udist/utime
    
    ginput : dict = jl.growth_rate_input_dict()
    s = data.axes[1][s_index]
    omega_cgs = astrounit2KeperianAngularVelocity(s,data.params['Origin_sink_mass'])
    ginput['Ω'] = omega_cgs
    try:
        cs = msetup["cs"]
    except KeyError:
        try:
            cs0 = msetup["cs0"]
        except KeyError:
            raise ValueError("Cannot determine the sound speed 'cs', Please assign in the 'msetup' as msetup['cs']")
        else:
            q = data.params['qfacdisc']
            cs = locally_isothermal_sound_speed(s,q,cs0)
            cs_cgs = cs * uv
    else:
        cs_cgs = cs
    ginput['cs'] = cs_cgs
    for key in data.column_names.keys():
        column = data.column_names[key]
        if 'St' in column:
            ginput['St'] = data.data_dict[key][s_index,phi_index]
            break
    for key in column_indices.keys():
        ginput[key] = data.data_dict[column_indices[key]][s_index,phi_index]
    ginput['vx'] /= cs_cgs
    ginput['vy'] /= cs_cgs
    ginput['ωx'] /= cs_cgs
    ginput['ωy'] /= cs_cgs
    growth_rate = jl.growth_rate_vectors(Κx=msetup['Kx'],
                                    Κz=msetup['Kz'],
                                    St=ginput['St'],
                                    ρg=ginput['ρg'],
                                    ρd=ginput['ρd'],
                                    vx=ginput['vx'],
                                    vy=ginput['vy'],
                                    ωx=ginput['ωx'],
                                    ωy=ginput['ωy'],)
    return growth_rate

def on_click(event,fax:LcartRpolar_plot,s:np.ndarray,phi:np.ndarray,data:PhantomRevealerAnalysisResult,msetup:dict,column_indices:dict,colormap_L):
    marker_label = 'polar_marker'
    if event.inaxes == fax.ax[1]:
        phi_click = event.xdata
        phi_index = value2closestvalueindex(phi,phi_click)
        s_click = event.ydata
        s_index = value2closestvalueindex(s,s_click)
        print(f"Clicked at phi: {phi_click}, r: {s_click}")
        print(f"Closest index: ({phi_index},{s_index}), Value: ({phi[phi_index]},{s[s_index]})\n")
        M_mass = data.params["Origin_sink_mass"]
        tslt = 3000
        minimum_growth_rate_per_omega = minimum_growth_rate(s[s_index],M_mass,tslt)
        print(f"For a spiral with lifetime={tslt} yr, the growth rate at r={s[s_index]} needs to larger then s/Ω={minimum_growth_rate_per_omega}.\n")
        
        for child in fax.ax[1].get_children():
            if hasattr(child, 'get_label') and child.get_label() == marker_label:
                child.remove()
                
        growth_rate = growth_rate_calculation(data,s_index,phi_index,msetup,column_indices)
        fax.ax[1].plot(phi[phi_index], s[s_index], 'o', color='r', label=marker_label)
        fax.clear_ax(0)
        fax.pcolor_draw(growth_rate,0,colormap_L,r"$s/\Omega$",True,"",[1e-4,0.1],False)
        fax.set_Lcart_label()
        fax.set_Lcart_scale('log')
        fax.fig.canvas.flush_events()
    return 0

def core(file:str,Polar_index:int) -> LcartRpolar_plot: 
    def on_switch(event):
        nonlocal j
        if event.key in 'ad':
            if event.key == 'a':
                j -= 1
            elif event.key == 'd':
                j += 1
            Polar_z = data.data_dict[Polar_indices[(Polar_Iindex+j)%len(Polar_indices)]]
            Polar_label = data.params["column_units"][Polar_indices[(Polar_Iindex+j)%len(Polar_indices)]]
            Polar_z = np.column_stack((Polar_z,Polar_z[:,0]))
            
            fax.clear_ax(1)
            fax.pcolor_draw(Polar_z,1,colormap_R,Polar_label,Polar_log,f"t = {time} yr",vlimr,False)
            fax.fig.canvas.flush_events()
    # ----------------Parameters setting----------------
    Kx = np.logspace(0.0,4.0,150)
    Kz = np.logspace(0.0,4.0,151)
    
    colormap_L = "jet"
    colormap_bottom2white = True
    colormap_R = "binary"
    Polar_log = False
    vlimr = None
    
    # Corresponding column indices
    rhog_index = 10
    rhod_index = 11
    vx_index = 12
    vy_index = 18
    omegax_index = 13
    omegay_index = 19
    
    # Manual setup some parameters
    msetup = {}
    msetup['cs0'] = 0.158113883                             # In Phantom code unit
    # --------------------------------------------------
    msetup['Kx'] = Kx
    msetup['Kz'] = Kz
    column_indices = {}
    column_indices["ρg"] = rhog_index
    column_indices["ρd"] = rhod_index
    column_indices["vx"] = vx_index
    column_indices["vy"] = vy_index
    column_indices["ωx"] = omegax_index
    column_indices["ωy"] = omegay_index
    
    # Modified colormap (If necessary)
    if colormap_bottom2white:
        colormap_L = colormap_with_base(colormap_L)
    
    j = 0
    
    data = PhantomRevealerAnalysisResult.Read_HDF5(filepath=file)
    if not data.params["Analysis_type"] == 'Faceon_disk':
        raise LookupError(f"The input file should have the analysis tag with 'Faceon_disk' rather then {data.params["Analysis_type"]}")
    data.transfer_cgs()
    data.add_dust2gas_ratio()
    data.add_St()
    data.add_vsub()
    Polar_indices = sorted(list(data.data_dict.keys()))
    Polar_Iindex = value2closestvalueindex(Polar_indices,Polar_index)
    s = data.axes[1]
    phi = data.axes[2]
    Polar_z = data.data_dict[Polar_indices[(Polar_Iindex+j)%len(Polar_indices)]]
    Polar_label = data.params["column_units"][Polar_indices[(Polar_Iindex+j)%len(Polar_indices)]]
    
    if not isclose(phi[-1],2*np.pi):
        phi = np.hstack((phi,2*np.pi))
        Polar_z = np.column_stack((Polar_z,Polar_z[:,0]))
    time = int(round(data.time))
    # Generate the window of figures 
    fax = LcartRpolar_plot(Kx,Kz,r"$k_{x}H$",r"$k_{z}H$", True, True,s,phi,r"$r$ [au]",r"$\phi$")
    fax.setup_fig((14,6))
    fax.__class__.anato_text_position = [-0.02,1.00]
    fax.pcolor_draw(Polar_z,1,colormap_R,Polar_label,Polar_log,f"t = {time} yr",vlimr,False)
    fax.fig.canvas.mpl_connect('button_press_event',  lambda event: on_click(event, fax=fax, s=s, phi=phi, data=data,msetup=msetup, column_indices=column_indices,colormap_L=colormap_L))
    fax.fig.canvas.mpl_connect('key_press_event',  on_switch)
    fax.draw_fig()
    return fax
    
def main(args) -> int:
    if args.latex:
        rcParams_update(font_family='Times New Roman', font_size=18, text_usetex=True, text_latex_preamble=r"\usepackage{amsfonts}")
    else:
        rcParams_update(font_family='sans-serif', font_size=18, text_usetex=False, text_latex_preamble="")
    try:
        fax = core(args.filepath,args.polar)
        while plt.fignum_exists(fax.fig.number):
            plt.pause(0.1)
    except KeyboardInterrupt:
        fax.close_fig()
        print("End Program.")
    return 0

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="The file input checker.")
    parser.add_argument(
        'filepath',
        type=str,
        help="Target file with Analysis_tag='Faceon_disk'."
    )
    parser.add_argument(
        '-p','--polar',
        type=int,
        default=2,
        help="The index of polar plot in file."
    )
    parser.add_argument(
        '-l', '--latex',
        action='store_true',
        help="Enable latex mode"
    )
    args = parser.parse_args()
    main(args)