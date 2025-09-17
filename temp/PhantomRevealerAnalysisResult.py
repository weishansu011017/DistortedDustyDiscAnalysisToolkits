import numpy as np
import h5py
import re
import os

def astrounit2KeperianAngularVelocity(r,M):
    r_cgs = 14959787069100.0 * r
    M_cgs =1.9891e33* M
    G = 6.67e-8
    Ω = np.sqrt((G*M_cgs)/(r_cgs**3))
    return Ω

def locally_isothermal_sound_speed(r,q,cs0):
    return cs0 * (r**(-q))

class PhantomRevealerAnalysisResult:
    '''
    The analysis data that genetared from PhantomRevealer.
    
    Field in Class:
        For data:
            analysis_type: Verify the type of analysis
            time: The time flag of file
            data_dict: The dictionary that stores all of the quantity (Dict(column_index(int) -> np.ndarray))
            axes: The axes of result array.
            column_names: The column name for each quantuty (Dict(column_index(int) -> str))
            params: Other parameters of simulation
    '''
    def __init__(self, time, data_dict, axes, column_names, params) -> None:
        self.time : np.float64 = time
        self.data_dict : dict = data_dict
        self.axes : dict = axes
        self.column_names : dict = column_names
        self.params : dict = params
        
    def transfer_cgs(self, year:bool = True):
        def replace_grident_exp(string):
            regex = r"cm\$\^\{(-?\d+)\}"
            m = re.search(regex, string)
            if m:
                exponent = int(m.group(1)) - 1
                new_exponent_str = f"cm$^{{{exponent}}}"
                new_str = string.replace(m.group(0), new_exponent_str)
                
                return new_str
            else:
                return string
        def extract_suffix(s):
            idx = s.rfind('_')
            if idx == -1:
                return ""
            else:
                result = s[idx+1:]
                if result.endswith("]"):
                    result = result[:-1]
                return result
        umass = self.params["umass"]
        udist = self.params["udist"]
        utime = self.params["utime"]
        try:
            _ = self.params["_cgs"]
        except KeyError:
            usigma = umass / (udist**2)
            urho = umass / (udist**3)
            uv = udist / utime
            if year:
                self.time *= (utime / 31536000)
                self.params["time"] *= (utime / 31536000)
            else:
                self.time *= utime
                self.params["time"] *= utime
            self.params["grainsize"] *= udist
            self.params["graindens"] *= urho
            column_unit = {}
            for key in self.column_names.keys():
                column_name = self.column_names[key]
                suffix = extract_suffix(column_name)
                
                if ("sigma" in column_name) or ("Sigma" in column_name):
                    self.data_dict[key] *= usigma
                    if "s_" in column_name:
                        header = r"$\Sigma_{s,"
                        suffix = suffix + r"}"
                        unit = r"$ [g cm$^{-2}$]"
                    elif "ϕ_" in column_name:
                        header = r"$\Sigma_{\phi,"
                        suffix = suffix + r"}"
                        unit = r"$ [g cm$^{-2}$]"
                    else:
                        header = r"$\Sigma_"
                        unit = r"$ [g cm$^{-2}$]"
                elif "rho" in column_name:
                    self.data_dict[key] *= urho
                    header = r"$\rho_"
                    unit = r"$ [g cm$^{-3}$]"
                elif "vs" in column_name:
                    self.data_dict[key] *= uv
                    header = r"$v_{s,"
                    suffix = suffix + r"}"
                    unit = r"$ [cm s$^{-1}$]"
                elif ("vphi" in column_name) or ("vϕ" in column_name):
                    self.data_dict[key] *= uv
                    if ("vphi_k" in column_name) or ("vϕ_k" in column_name):
                        header = r"$v_{\phi,"
                        suffix = suffix + r"-k}"
                        unit = r"$ [cm s$^{-1}$]"
                    else:
                        header = r"$v_{\phi,"
                        suffix = suffix + r"}"
                        unit = r"$ [cm s$^{-1}$]"
                elif "vz" in column_name:
                    self.data_dict[key] *= uv
                    header = r"$v_{z,"
                    suffix = suffix + r"}"
                    unit = r"$ [cm s$^{-1}$]"
                else:
                    header = column_name
                    suffix = r""
                    unit = r""
                column_unit[key] = header + suffix + unit
                if "∇" in column_name:
                    self.data_dict[key] /= udist
                    if "⋅" in column_name:
                        column_unit[key] = r"$\nabla\cdot$" + replace_grident_exp(column_unit[key])
                    elif "×" in column_name:
                        column_unit[key] = r"$\nabla\times$" + replace_grident_exp(column_unit[key])
                    else:
                        column_unit[key] = r"$\nabla$" + replace_grident_exp(column_unit[key])
                self.params["column_units"] = column_unit.copy()
        finally:
            self.params["_cgs"] = True
    
    def add_more_label(self, column_index:int, label:str):
        try:
            current_column_unit : dict = self.params["column_units"]
        except KeyError:
            raise LookupError("You should calling the transfer_cgs() method before calling this method!")
        if column_index in current_column_unit.keys():
            while True:
                yn = input(f"Old column unit {current_column_unit[column_index]} has found. Are you sure you want to replace it? [y/n]")
                if yn == "y":
                    break
                elif yn == "n":
                    return
                else:
                    pass
        self.params["column_units"][column_index] = label
        
    def Check_array_quantities(self, array_index:int):
        if "_cgs" in self.params.keys():
            In_cgs = self.params["_cgs"]
        else:
            In_cgs = False
        array : np.ndarray = self.data_dict[array_index]
        column_name : str = self.column_names[array_index]
        shape = array.shape
        arrmax = np.nanmax(array)
        arrmin = np.nanmin(array)
        arraverage = np.nanmean(array)
        arrmedian = np.nanmedian(array)
        arrstd = np.nanstd(array)
        print(f"--------------Properties of array {column_name}--------------")
        print(f"Size: {shape}")
        print(f"In cgs unit?: {In_cgs}")
        print(f"Maximum: {arrmax}")
        print(f"Minimum: {arrmin}")
        print(f"Average: {arraverage}")
        print(f"Median: {arrmedian}")
        print(f"STD: {arrstd}")
        print("----------------------------------------------------------------")
    
    @staticmethod
    def Read_HDF5(filepath): 
        def read_dict(f, name, K=str):
            dictionary = {}
            g = f[name]
            for key,val in g.items():
                if isinstance(val[()], np.bytes_):
                    dictionary[K(key)] = val[()].decode('utf-8')
                elif isinstance(val[()], np.ndarray):
                    dictionary[K(key)] = val[()].copy().T      # Trasepose because of the storage system
                else:
                    dictionary[K(key)] = val[()]
            return dictionary 
            
        with h5py.File(filepath, 'r') as f:
            time = f['time'][()]
            data_dict = read_dict(f,'data_dict',int)
            axes = read_dict(f, 'axes', int)
            column_names = read_dict(f, 'column_names', int)
            params = read_dict(f, 'params', str)
            
            data = PhantomRevealerAnalysisResult(time, data_dict, axes, column_names, params)
            return data
        
    def add_dust2gas_ratio(self, column_index= 61):
        self.transfer_cgs()
        rhog = None
        rhod = None
        for key in self.column_names.keys():
            column_name = self.column_names[key]
            if ("rhom_g" in column_name) or ("rho_g" in column_name):
                rhog = self.data_dict[key]
            elif ("rhom_d" in column_name) or ("rho_d" in column_name):
                rhod = self.data_dict[key]
        if (rhog is None) or (rhod is None):
            raise LookupError("The density column has not found!")
        d2g = rhod/rhog
        self.column_names[column_index] = f"[{column_index}  dust-to-gas]"
        self.add_more_label(column_index,r"$\varepsilon$")
        self.data_dict[column_index] = d2g
        
    def add_St(self, column_index= 62):
        for key in self.column_names.keys():
            column = self.column_names[key]
            if 'St' in column:
                return
        self.transfer_cgs()
        Sigmag = None
        for key in self.column_names.keys():
            column_name = self.column_names[key]
            if ("Sigmam_g" in column_name) or ("Sigma_g" in column_name):
                Sigmag = self.data_dict[key]
        if (Sigmag is None):
            raise LookupError("The surface density column has not found!")
        a_s = self.params['grainsize']
        rho_s = self.params['graindens']
        St = (np.pi/2)*(a_s*rho_s)/Sigmag
        self.column_names[column_index] = f"[{column_index}  St]"
        self.add_more_label(column_index,r"St")
        self.data_dict[column_index] = St
        
    def add_vsub(self, column_index = 63, use_subkep_vphi=True):
        for column in self.column_names.values():
            if "vsub" in column:
                return
        self.transfer_cgs()
        vsg = None
        vsd = None
        vϕg = None
        vϕd = None
        vzg = None
        vzd = None
        
        if use_subkep_vphi:
            vphi_name1 = "vϕ-vϕ_k"
            vphi_name2 = "vϕ-vϕ_km"
        else:
            vphi_name1 = "vϕ"
            vphi_name2 = "vϕm"
        
        for key in self.column_names.keys():
            column = self.column_names[key]
            if ("vs_g" in column) or ("vsm_g" in column):
                vsg = self.data_dict[key]
            elif ("vs_d" in column) or ("vsm_d" in column):
                vsd = self.data_dict[key]
            elif (vphi_name1+"_g" in column) or (vphi_name2+"_g" in column):
                vϕg = self.data_dict[key]
            elif (vphi_name1+"_d" in column) or (vphi_name2+"_d" in column):
                vϕd = self.data_dict[key]
            elif ("vz_g" in column) or ("vzm_g" in column):
                vzg = self.data_dict[key]
            elif ("vz_d" in column) or ("vzm_d" in column):
                vzd = self.data_dict[key]
        if (vsg is None) or (vsd is None) or (vϕg is None) or (vϕd is None):
            raise KeyError("The velocity column has not been found!")
        vs = np.abs(vsg-vsd)
        vϕ = np.abs(vϕg-vϕd)
        if (vzg is None) or (vzd is None):
            vsub = np.sqrt(vs**2 + vϕ**2)
        else:
            vz = np.abs(vzg-vzd)
            vsub = np.sqrt(vs**2 + vϕ**2 + vz**2)
        self.column_names[column_index] = f"[{column_index} vsub]"
        self.add_more_label(column_index,r"$| \mathbf{v}_g - \mathbf{v}_d |$")
        self.data_dict[column_index] = vsub