import subprocess
import itertools
import matplotlib.artist
import numpy as np
import matplotlib.font_manager
import matplotlib.figure as mfg
import matplotlib.axes as maxe
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as mcolors
import matplotlib.gridspec as gspec

# Set nan if divided by zero or taking log for negative number.
np.seterr(divide='ignore',invalid='ignore')

def value2closestvalueindex(array, target):
    array = np.asarray(array)
    target_index = np.argmin(np.abs(array - target))
    return target_index

def replace_inf_with_nan(arr):
    mask = np.isinf(arr)
    arr[mask] = np.nan
    return arr

def check_latex_installed():
    try:
        subprocess.run(['latex', '--version'], check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        return True
    except (subprocess.CalledProcessError, FileNotFoundError):
        return False

def rcParams_update(font_family='Times New Roman', font_size=13, text_usetex=True, text_latex_preamble=r"\usepackage{amsfonts}", savefig_dpi=450,savefig_format="eps"):
    '''
    Update the font setting
    '''
    plt.rcParams.update({
    "font.family": font_family,
    "font.size": font_size,
    "text.usetex": text_usetex,
    "text.latex.preamble": text_latex_preamble,
    "savefig.dpi": savefig_dpi,
    "savefig.format": savefig_format
})
    
def Current_rcParams():
    return plt.rcParams
    
# Update LaTeX Rendering
if check_latex_installed():
    for file in matplotlib.font_manager.findSystemFonts(fontpaths=None, fontext='ttf'):
        if 'Time' in file:
            rcParams_update()
            break
        else:
            rcParams_update('sans-serif')
else:
    rcParams_update('sans-serif', 13, False,'')

def openinteractive():
    '''
    Open the interactive interface for plotting
    '''
    if not plt.isinteractive():
        plt.ion()
            
def closeinteractive():
    '''
    Close the interactive interface for plotting
    '''
    if plt.isinteractive():
        plt.ioff()
        
def Get_vminmax(array:np.ndarray):
    '''
    Calculate the minimum and maximum value for colorbar.
    '''
    array = replace_inf_with_nan(array)
    median = np.nanmedian(array)
    std = np.nanstd(array)
    vmax = median + 3*std
    vmin = median - 3*std
    if (np.nanmin(array)>=0.0) and (vmin<0.0):
        if vmax<1e-15:
            vmin = 5e-18
        else:
            vmin = 1e-15
        
    print(f"Warning: Automatically calculate (vmin, vmax) = {(vmin,vmax)}")
    return np.array([vmin,vmax])
        
def polar2cart(theta:float, radius:float):
    '''
    Transfer the polar coordinate into the cartisian coordinate.
    '''
    x = radius * np.cos(theta)
    y = radius * np.sin(theta)
    return x, y

def Basic_properties_array(arr,array_name):
    max = np.nanmax(arr)
    min = np.nanmin(arr)
    mean = np.nanmean(arr)
    median = np.nanmedian(arr)
    std = np.nanstd(arr)
    print(f'----------------Properties of Array {array_name}----------------')
    print(f'Maximum: {max}')
    print(f'Minimum: {min}')
    print(f'Mean: {mean}')
    print(f'Median: {median}')
    print(f'STD: {std}')
    print('----------------------------------------------------------------')

def color_cycle():
    return itertools.cycle(plt.rcParams['axes.prop_cycle'].by_key()['color'])

def colormap_with_base(cmap_name, n_colors=256, to_white=True, white_ratio=0.05):
    """
    Create a new colormap that smoothly transitions to white or black in the low-value region.
    (Note: Fully written by ChatGPT. I am so appreciate to the power of AI lol.)

    Parameters:
    - cmap_name (str): Name of the built-in colormap in Matplotlib.
    - n_colors (int): Number of colors in the colormap, default is 256.
    - to_white (bool): If True, transitions to white; if False, transitions to black.
    - white_ratio (float): Ratio of the white or black region in the colormap (0~1), default is 0.05 (5%).

    Returns:
    - new_cmap: A colormap with a white or black gradient in the low-value region.
    """
    # Get the specified colormap using the updated API for Matplotlib 3.7+
    base_cmap = plt.get_cmap(cmap_name, n_colors)
    
    # Define base color: white or black
    base_color = np.array((1, 1, 1, 1)) if to_white else np.array((0, 0, 0, 1))
    
    # Get the color at the low end of the original colormap
    start_color = np.array(base_cmap(0))  # Bottom color of the colormap

    # Calculate the number of colors in the white/black region
    n_white = int(n_colors * white_ratio)

    # Generate transition colors from white/black to the colormap's start color
    transition_colors = [(1 - t) * base_color + t * start_color 
                         for t in np.linspace(0, 1, n_white)]
    
    # Combine transition colors with the rest of the colormap
    colors = transition_colors + [base_cmap(i) for i in range(n_white, n_colors)]

    # Create a new colormap with a smooth gradient
    new_cmap = mcolors.LinearSegmentedColormap.from_list(f"{cmap_name}_with_base", colors)
    
    return new_cmap

class figure_ax:
    '''
    The plotting object in the code.
    
    Field in Class:
        fig: The figure/Canvas.
        ax: The axes of plotting/colorbar
        proj: The projection of plotting
        
    Usage:
        1. Generate an object
            fax = figure_ax()
            
        2. Setup the desired figure
            fax.setup_fig(ncols=2, nrows=2, figsize=(10,6))
            
        3. Make the plot at each axes
            fax.ax[0].plot(x,y,*plotting_params)
            cont = fax.ax[1].pcolor(x,y,z,*plotting_params)
            
        4. Setup the colorbar for each map (if necessary). Allowing giving the colorbar a specific label so that the new colorbar can be drawn on the old ax rather then the new one.
            colorbar = fax.setup_colorbar(cont,'LABEL_OF_COLORBAR')
            
        5. Save the figure
            fax.fig.savefig(filepath, dpi=dpi,bbox_inches='tight',transparent=False)
        
        6. Close the figure
            fax.close_fig()

        
    '''
    def __init__(self,proj=None):
        '''
        Two kinds of projection
        "None": Normal cartisian plotting
        "polar": Polar plotting
        "3d": 3D plotting
        ""
        '''
        self.fig: mfg.Figure = None
        self.ax = None
        self.ncols = None
        self.nrows = None
        if (not isinstance(proj,list)) and (not isinstance(proj,str)) and (not proj is None):
            raise ValueError("The 'proj' argument should be in either str, list or None")
        else:
            if isinstance(proj,list):
                for p in proj:
                    if (not isinstance(p,str)) and (not p is None):
                        raise ValueError("All the element in 'proj' with type 'list' should be in str or None")
            self.proj = proj
        
    def checking_initialized(self):
        if self.fig is None or self.ax is None:
            raise NameError(f"No figure or ax has found! You should initialized the figure by the 'setup_figure()' method.")
        
    def setup_fig(self, nrows=1,ncols=1,figsize = (12, 8)):
        openinteractive()
        if self.fig is None or self.ax is None:
            self.ncols = ncols
            self.nrows = nrows
            self.fig = plt.figure(figsize=figsize, constrained_layout=True)
            gs = gspec.GridSpec(nrows, ncols, figure=self.fig, hspace=0.0, wspace=0.0)

            ax_list = []
            for r in range(nrows):
                for c in range(ncols):
                    if isinstance(self.proj,list):
                        projection = self.proj[r*ncols + c]
                        sharex = None
                        sharey = None
                    else:
                        projection = self.proj
                        if ncols > 1 and nrows > 1:
                            sharex = 'all'
                            sharey = 'all'
                        elif ncols > 1:
                            sharex = None
                            sharey = 'all'
                        elif nrows > 1:
                            sharex = 'all'
                            sharey = None
                        else:
                            sharex = None
                            sharey = None

                    if c == 0 and r == 0:
                        ax = self.fig.add_subplot(gs[r, c], projection=projection)
                    else:
                        ax = self.fig.add_subplot(gs[r, c], projection=projection, sharex=ax_list[0] if sharex else None, sharey=ax_list[0] if sharey else None)
                    ax_list.append(ax)
                    
                    if projection == 'polar':
                        ax.set_xticklabels([]) 
                        ax.set_yticklabels([])  
                        ax.xaxis.set_visible(False)  
                        ax.yaxis.set_visible(False)
                    elif projection is None:
                        if c != 0:  
                            plt.setp(ax.get_yticklabels(), visible=False)
                        
                        if r != nrows - 1:  
                            plt.setp(ax.get_xticklabels(), visible=False)
                    elif projection == '3d':
                        pass
                        

            self.ax = ax_list[0] if (ncols == 1 and nrows == 1) else ax_list
        
            if (not isinstance(self.proj,list)) and (self.proj != 'polar'):
                plt.tight_layout()
                
    def get_number_of_ax(self):
        ax = self.ax
        if not isinstance(ax, list):
            return 1
        else:
            return len(self.ax)
    
    def is_same_column(self,axid1, axid2):
        ncols = self.ncols
        if axid1%ncols == axid2%ncols:
            return True
        else:
            return False
        
    def is_same_row(self,axid1,axid2):
        ncols = self.ncols
        if axid1//ncols == axid2//ncols:
            return True
        else:
            return False
    
    def setup_colorbar_ax(self, axids=0, colorbar_ax_label=None, fraction=0.05, pad=0.04):
        if colorbar_ax_label is None:
            colorbar_ax_label = f"<colorbarAX>"
            print(f"The label of new axis of colobar is set to be {colorbar_ax_label}. The spcified label would be recommand.")
        if self.get_number_of_ax() == 1:
            print("The function 'setup_colorbar_ax()' only valid for the multiplot. For a single plot, you won't need this function.")
            return
        if isinstance(axids,list):
            for i in range(len(axids)-1):
                axid1 = axids[i]
                axid2 = axids[i+1]
                if not (self.is_same_column(axid1,axid2)):
                    raise IndexError("All of the axes should be located on the same column!")
        else:
            axids = [axids]
        fig = self.fig
        ax = self.ax
        top_ax_pos = ax[axids[0]].get_position().bounds
        bottom_ax_pos = ax[axids[-1]].get_position().bounds
        colorbar_height = top_ax_pos[1] + top_ax_pos[3] - bottom_ax_pos[1]
        if colorbar_height <= 0:
            raise ValueError("Calculated colorbar height is non-positive. Check the subplot ordering.")
        colorbar_width = top_ax_pos[2] * fraction
        colorbar_left = top_ax_pos[0] + top_ax_pos[2] + pad
        colorbar_bottom = bottom_ax_pos[1]
        colorbar_ax = fig.add_axes([colorbar_left, colorbar_bottom, colorbar_width, colorbar_height])
        colorbar_ax.set_label(colorbar_ax_label)
        return colorbar_ax
            
    def setup_colorbar(self,cont,colorbar_ax_label = None):
        ''' Setup the colobar for the specific heatmap.
        
        The heatmap can be selected by specifing the "colorbar_ax_label" variable.
        '''
        if colorbar_ax_label is None:
            colarbar_ax = None
            for ax in self.fig.axes:
                if ax.get_label() == '<colorbar>':
                    colarbar_ax = ax
            if colarbar_ax is None:
                colorbar = self.fig.colorbar(cont)
            else:
                colorbar = self.fig.colorbar(cont,cax = colarbar_ax)
        else:
            colarbar_ax = None
            for ax in self.fig.axes:
                if ax.get_label() == colorbar_ax_label:
                    colarbar_ax = ax
            if colarbar_ax is None:
                colorbar = self.fig.colorbar(cont)
                colorbar.ax.set_label(colorbar_ax_label)
            else:
                colorbar = self.fig.colorbar(cont,cax = colarbar_ax)
        return colorbar
    
    def clear_ax(self,ax_index):
        if self.get_number_of_ax() > 1:
            self.ax[ax_index].clear()
        else:
            self.ax.clear()
    
    def draw_fig(self):
        if not self.fig is None:
            self.fig.canvas.draw()
            self.fig.canvas.flush_events()
    
    def save_fig(self,filepath='figure_1.png',dpi=450):
        if self.fig:
            self.fig.savefig(filepath, dpi=dpi,bbox_inches='tight',transparent=False)
        else:
            print("No figure to save.")
    
    def close_fig(self):
        if self.fig is not None and self.ax is not None:
            plt.close(self.fig) 
            self.fig = None
            self.ax = None
        
    def reset_fig(self):
        ncols = self.ncols
        nrows = self.nrows
        figsize = tuple(self.fig.get_size_inches())
        self.close_fig()
        self.setup_fig(nrows,ncols,figsize)
        
    def reordered_legend(self, axid=0,ncol=2,order=[0, 2, 4, 6, 1, 3, 5, 7],loc='best'):
        if isinstance(self.ax,list):
            ax = self.ax[axid]
        else:
            ax = self.ax
        handles, labels = ax.get_legend_handles_labels()
        if (not len(handles)==len(order)):
            return
        else:
            handles = [handles[i] for i in order]
            labels = [labels[i] for i in order]
            if ax.get_legend() is None:
                ax.legend()
            ax.get_legend().remove()
            ax.legend(handles, labels,loc=loc, ncol=ncol)
            
class two_axes_plot(figure_ax):
    props = dict(boxstyle='round', facecolor='black')
    anato_text_position=(0.00,0.95)
    def __init__(self, x:np.ndarray, y:np.ndarray, xlabel:str, ylabel:str, proj=None):
        super().__init__(proj)
        self._x : np.ndarray = np.array(x)
        self._y : np.ndarray = np.array(y)
        self._xlabel : str = xlabel
        self._ylabel : str = ylabel
        self.meshgrid()
    
    def grid_shape(self):
        return (self._y.shape[0], self._x.shape[0])
    
    def meshgrid(self):
        self._X,self._Y = np.meshgrid(self._x, self._y)
        
    def get_colorbar_pos_indies(self):
        indies = np.zeros(self.nrows)
        for i in range(0,self.nrows):
            indies[i] = (i+1)*self.ncols-1
        return indies
    
    def pcolor_draw(self,images:list,colormaps:list, clabels:list, Log_flags:list, anatonate_labels:list, vlims = None,draw=True):
        '''
        The clims parameters should be the following format:
        
        [[vmin1,vmax1], [vmin2,vmax2], [vmin3,vmax3], ... ]
        
        '''
        cls = self.__class__
        # Checking whether the number of figures is matching.
        self.checking_initialized()
        if len(images) != self.get_number_of_ax():
            raise ValueError(f"Mismatching number of arrays! The number of figures was set to be {self.get_number_of_ax()}, but {len(images)} elements in images was given.")
        if len(colormaps) != self.nrows:
            raise ValueError(f"Mismatching number of colormaps! The number of rows of figures was set to be {self.nrows}, but {len(colormaps)} elements in colormaps was given.")
        if len(Log_flags) != self.nrows:
            raise ValueError(f"Mismatching number of Log_flags! The number of rows of figures was set to be {self.nrows}, but {len(Log_flags)} elements in Log_flags was given.")
        if len(clabels) != self.nrows:
            raise ValueError(f"Mismatching number of clabels! The number of rows of figures was set to be {self.nrows}, but {len(clabels)} elements in clabels was given.")
        if len(anatonate_labels) != self.ncols:
            raise ValueError(f"Mismatching number of anatonate_labels! The number of columns of figures was set to be {self.ncols}, but {len(anatonate_labels)} elements in anatonate_labels was given.")
        
        # Preparing Color limit
        colorbar_pos_indies = self.get_colorbar_pos_indies()
        if vlims is None:
            vlims = np.empty(self.nrows, dtype=np.ndarray)
            j = 0
            for i in range(self.get_number_of_ax()):
                if i in colorbar_pos_indies:
                    image = images[i]
                    vlims[j] = Get_vminmax(image)
                    j += 1
        else:
            if len(vlims) != self.nrows:
                raise ValueError(f"Mismatching number of colormap limits! The number of rows of figures was set to be {self.nrows}, but {len(vlims)} pairs of colormap limit was given.")
            else:
                for vlim in vlims:
                    if len(vlim) != 2:
                        raise ValueError(f"Wrong format of 'vlims' parameters! The clims parameters should be [[vmin1,vmax1], [vmin2,vmax2], [vmin3,vmax3], ... ]")
            
        # Preparing Color Norm
        Norms = np.empty(self.nrows, dtype=object)
        for j in range(len(Log_flags)):
            vmin, vmax = vlims[j]
            if Log_flags[j]:
                if vmin < 0 and vmax > 0: 
                    linthresh = min(abs(vmin), abs(vmax)) * 0.1
                    Norms[j] = mcolors.SymLogNorm(linthresh=linthresh, vmin=vmin, vmax=vmax)
                else:
                    Norms[j] = mcolors.LogNorm(vmin=vmin, vmax=vmax)
            else: 
                Norms[j] = mcolors.Normalize(vmin=vmin, vmax=vmax)
        
        # Preparing interior colormap ax labels
        cax_label = [f"<colorbar_col{n+1}>" for n in range(self.nrows)]
        # Preparing grid
        X = self._X
        Y = self._Y
        ax = self.ax
        # Make sure the axis has its minimun while plotting
        x = self._x
        y = self._y
        xlim = [x[0],x[-1]]
        ylim = [y[0],y[-1]]
        # Checking the grid size is mathing, and then drawing.
        j = 0
        for i,image in enumerate(images):
            image = np.array(image)
            k = i % self.ncols
            if image.shape != self.grid_shape():
                raise ValueError(f"Mismatching shape of array! The grid size of figures was set to be ({self.grid_shape()}), but {image.shape} was given.")
            else:
                if isinstance(ax, list):
                    cont = ax[i].pcolor(X,Y,image, cmap=colormaps[j], norm=Norms[j])
                    ax[i].text(*cls.anato_text_position,anatonate_labels[k], horizontalalignment='left',transform=ax[i].transAxes,c='white', fontsize=plt.rcParams["font.size"], verticalalignment='top', bbox=cls.props)
                    if i in colorbar_pos_indies:
                        colorbar = self.setup_colorbar(cont,colorbar_ax_label=cax_label[j])
                        colorbar.set_label(clabels[j])
                        j += 1
                    ax[i].set_xlim(xlim[0],xlim[1])
                    ax[i].set_ylim(ylim[0],ylim[1])
                    ax[i].set_rasterized(True)
                else:
                    cont = ax.pcolor(X,Y,image, cmap=colormaps[0], norm=Norms[0])
                    ax.text(*cls.anato_text_position,anatonate_labels[0], horizontalalignment='left', transform=ax.transAxes,c='white', fontsize=plt.rcParams["font.size"], verticalalignment='top', bbox=cls.props)
                    colorbar = self.setup_colorbar(cont,colorbar_ax_label=cax_label[0])
                    colorbar.set_label(clabels[0])
                    ax.set_xlim(xlim[0],xlim[1])
                    ax.set_ylim(ylim[0],ylim[1])
                    ax.set_rasterized(True)
                    
        if draw:
            self.draw_fig()
            
    def streamplot_draw(self,vxgrids:list,vzgrids:list,color="black",density=2.0,draw=True):
        self.checking_initialized()
        if len(vxgrids) != self.get_number_of_ax():
            raise ValueError(f"Mismatching number of arrays! The number of figures was set to be {self.get_number_of_ax()}, but {len(vxgrids)} elements in images was given.")
        if len(vzgrids) != self.get_number_of_ax():
            raise ValueError(f"Mismatching number of arrays! The number of figures was set to be {self.get_number_of_ax()}, but {len(vzgrids)} elements in images was given.")
        X = self._X
        Y = self._Y
        ax = self.ax
        
        for i in range(self.get_number_of_ax()):
            vxgrid = np.array(vxgrids[i])
            vzgrid = np.array(vzgrids[i])
            
            if vxgrid.shape != self.grid_shape():
                raise ValueError(f"Mismatching shape of array! The grid size of figures was set to be ({self.grid_shape()}), but {vxgrid.shape} was given.")
            elif vzgrid.shape != self.grid_shape():
                raise ValueError(f"Mismatching shape of array! The grid size of figures was set to be ({self.grid_shape()}), but {vzgrid.shape} was given.")
            else:
                if isinstance(ax, list):
                    ax[i].streamplot(X,Y,vxgrid,vzgrid,color=color, density=density)
                    ax[i].set_rasterized(True)
                else:
                    ax.streamplot(X,Y,vxgrid,vzgrid,color=color, density=density)
                    ax.set_rasterized(True)
        if draw:
            self.draw_fig()
        
class polar_plot(two_axes_plot):
    anato_text_position=(0.00,0.95)
    def __init__(self, s: np.ndarray, phi: np.ndarray, slabel: str, philabel: str):
        super().__init__(phi, s, philabel, slabel,  'polar')

    @property
    def s(self):
        return self._y
    
    @property
    def phi(self):
        return self._x
    
    @property
    def slabel(self):
        return self._ylabel
    
    @property
    def philabel(self):
        return self._xlabel
    
    @property
    def S(self):
        return self._Y
    
    @property
    def PHI(self):
        return self._X
    
    def pcolor_draw(self, images: list, colormaps: list, clabels: list, Log_flags: list, anatonate_labels: list, vlims=None, draw=True):
        super().pcolor_draw(images, colormaps, clabels, Log_flags, anatonate_labels, vlims, draw)
        ax = self.ax
        if isinstance(ax, list):
            for i in range(self.get_number_of_ax()):
                ax[i].set_rmin(-1)
        else:
            ax.set_rmin(-1)
                
        
            
class cart_plot(two_axes_plot):
    anato_text_position=(0.02,0.98)
    def __init__(self, x: np.ndarray, y: np.ndarray, xlabel: str, ylabel: str):
        super().__init__(x, y, xlabel, ylabel)
    
    @property
    def y(self):
        return self._y
    
    @property
    def x(self):
        return self._x
    
    @property
    def ylabel(self):
        return self._ylabel
    
    @property
    def xlabel(self):
        return self._xlabel
    
    @property
    def Y(self):
        return self._Y
    
    @property
    def X(self):
        return self._X
    
    def set_ylabel(self,axid):
        if self.get_number_of_ax() > 1:
            self.ax[axid].set_ylabel(self.ylabel)
        else:
            self.ax.set_ylabel(self.ylabel)
    
    def set_xlabel(self,axid):
        if self.get_number_of_ax() > 1:
            self.ax[axid].set_xlabel(self.xlabel)
        else:
            self.ax.set_xlabel(self.xlabel)
            
    def set_yscale(self,axid,scale='linear'):
        if self.get_number_of_ax() > 1:
            self.ax[axid].set_yscale(scale)
        else:
            self.ax.set_yscale(scale)
    
    def set_xscale(self,axid,scale='linear'):
        if self.get_number_of_ax() > 1:
            self.ax[axid].set_xscale(scale)
        else:
            self.ax.set_xscale(scale)
        
class LcartRpolar_plot(figure_ax):
    props = dict(boxstyle='round', facecolor='black')
    anato_text_position=(0.00,0.95)
    def __init__(self, x, y, xlabel, ylabel, logx, logy, s, phi, slabel, philabel):
        super().__init__(proj=[None, 'polar'])
        self.x : np.ndarray = x
        self.y : np.ndarray = y
        self.s : np.ndarray = s
        self.phi : np.ndarray = phi
        self.xlabel : str = xlabel
        self.ylabel : str = ylabel
        self.slabel : str = slabel
        self.philabel : str = philabel
        self.logx : bool = logx
        self.logy : bool = logy
        self.meshgrid()
        
    def meshgrid(self):
        self.X,self.Y = np.meshgrid(self.x, self.y)
        self.PHI,self.S = np.meshgrid(self.phi, self.s)
    
    def grid_shape(self,ax_index):
        if ax_index == 0:
            return (self.y.shape[0], self.x.shape[0])
        elif ax_index == 1:
            return (self.s.shape[0], self.phi.shape[0])
        else:
            raise IndexError("Only index 0 or 1 is allowed!")
    
    def call_grids(self,ax_index):
        if ax_index == 0:
            return self.X,self.Y
        elif ax_index == 1:
            return self.PHI,self.S
        else:
            raise IndexError("Only index 0 or 1 is allowed!")
    
    def call_axes(self,ax_index):
        if ax_index == 0:
            return self.x,self.y
        elif ax_index == 1:
            return self.phi,self.s
        else:
            raise IndexError("Only index 0 or 1 is allowed!")
    
    def setup_fig(self,figsize=(12, 7)):
        return super().setup_fig(1, 2, figsize)
    
    def reset_fig(self):
        figsize = tuple(self.fig.get_size_inches())
        self.close_fig()
        self.setup_fig(figsize)
        
    def set_Lcart_label(self):
        self.ax[0].set_ylabel(self.ylabel)
        self.ax[0].set_xlabel(self.xlabel)
        
    def set_Lcart_scale(self,scale='linear'):
        self.ax[0].set_yscale(scale)
        self.ax[0].set_xscale(scale)
    
    def pcolor_draw(self, image, ax_index:int, colormap:str, clabel:str, Log_flag:bool, anatonate_label:str, vlim = None,draw=True):
        cls = self.__class__
        image = np.array(image)
        if image.shape != self.grid_shape(ax_index):
            raise IndexError(f"Mismatching array size! The allowed array size {self.grid_shape(ax_index)} didn't match the size of 'z' {image.shape}!")
        grids = self.call_grids(ax_index)
        
        # Preparing Color limit
        if vlim is None:
            vlim = Get_vminmax(image)
        else:
            if len(vlim) != 2:
                raise ValueError(f"Wrong format of 'vlims' parameters! The clims parameters should be [[vmin1,vmax1], [vmin2,vmax2], [vmin3,vmax3], ... ]")
            
        # Preparing Color Norm
        Norm = mcolors.LogNorm(*vlim) if Log_flag else mcolors.Normalize(*vlim) 
        
        # Preparing interior colormap ax labels
        cax_label = "<colorbar_L>" if ax_index == 0 else "<colorbar_R>"
        
        # Make sure the axis has its minimun while plotting
        x,y = self.call_axes(ax_index)
        xlim = [x[0],x[-1]]
        ylim = [y[0],y[-1]]
        
        ax = self.ax[ax_index]
        cont = ax.pcolor(*grids,image, cmap=colormap, norm=Norm)
        ax.text(*cls.anato_text_position,anatonate_label, horizontalalignment='left',transform=ax.transAxes,c='white', fontsize=plt.rcParams["font.size"], verticalalignment='top', bbox=cls.props)
        colorbar = self.setup_colorbar(cont,colorbar_ax_label=cax_label)
        colorbar.set_label(clabel)
        ax.set_xlim(xlim[0],xlim[1])
        ax.set_ylim(ylim[0],ylim[1])
        if (ax_index == 0) and (self.logx):
            ax.set_xscale('log')
        if (ax_index == 0) and (self.logy):
            ax.set_yscale('log')
        if (ax_index == 1):
            ax.set_rmin(-1)
        
        if ax_index == 0: 
            ax.set_rasterized(True)
        if draw:
            self.draw_fig()

