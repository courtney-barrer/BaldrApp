import numpy as np 
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from mpl_toolkits.axes_grid1 import make_axes_locatable
import datetime 
from astropy import units as u
from matplotlib.widgets import Slider
import matplotlib.animation as animation
import math
from configparser import ConfigParser
from types import SimpleNamespace

def ini_to_namespace(ini_file):
    # convert ini file to python namespace
    
    # Create a ConfigParser object and read the INI file
    config = ConfigParser()
    config.read(ini_file)

    # Initialize an empty SimpleNamespace
    namespace = SimpleNamespace()

    # Iterate through sections and keys to populate the namespace
    for section in config.sections():
        section_namespace = SimpleNamespace()
        for key, value in config.items(section):
            setattr(section_namespace, key, value)
        
        # Set the section as an attribute of the main namespace
        setattr(namespace, section, section_namespace)

    return namespace


from configparser import ConfigParser
from types import SimpleNamespace

def ini_to_namespace(ini_file):
    # Create a ConfigParser object and read the INI file
    config = ConfigParser()
    config.read(ini_file)

    # Initialize an empty SimpleNamespace
    namespace = SimpleNamespace()

    # Iterate through sections and keys to populate the namespace
    for section in config.sections():
        section_namespace = SimpleNamespace()
        for key, value in config.items(section):
            # Try automatic type conversion using ConfigParser methods
            if config.has_option(section, key):
                # First attempt to convert to an integer
                try:
                    converted_value = config.getint(section, key)
                except ValueError:
                    # If it's not an int, try converting to a float
                    try:
                        converted_value = config.getfloat(section, key)
                    except ValueError:
                        # If it's not a float, check if it's a boolean
                        try:
                            converted_value = config.getboolean(section, key)
                        except ValueError:
                            # Fallback to original string if no conversion works
                            converted_value = value
            
            setattr(section_namespace, key, converted_value)
        
        # Set the section as an attribute of the main namespace
        setattr(namespace, section, section_namespace)

    return namespace



def get_DM_command_in_2D(cmd,Nx_act=12):
    # function so we can easily plot the DM shape (since DM grid is not perfectly square raw cmds can not be plotted in 2D immediately )
    #puts nan values in cmd positions that don't correspond to actuator on a square grid until cmd length is square number (12x12 for BMC multi-2.5 DM) so can be reshaped to 2D array to see what the command looks like on the DM.
    corner_indices = [0, Nx_act-1, Nx_act * (Nx_act-1), Nx_act*Nx_act]
    cmd_in_2D = list(cmd.copy())
    for i in corner_indices:
        cmd_in_2D.insert(i,np.nan)
    return( np.array(cmd_in_2D).reshape(Nx_act,Nx_act) )


def insert_concentric(smaller_array, larger_array):
    # Get the shapes of both arrays
    N, M = smaller_array.shape
    P, Q = larger_array.shape

    # Check if the smaller array can fit in the larger array
    if N > P or M > Q:
        raise ValueError("Smaller array must have dimensions less than or equal to the larger array.")

    # Find the starting indices to center the smaller array in the larger array
    start_row = (P - N) // 2
    start_col = (Q - M) // 2

    # Create a copy of the larger array to avoid modifying the input directly
    result_array = larger_array.copy()

    # Insert the smaller array into the larger array
    result_array[start_row:start_row + N, start_col:start_col + M] = smaller_array

    return result_array



def crop_pupil(pupil, image):
    """
    Detects the boundary of a pupil in a binary mask (with pupil = 1 and background = 0)
    and crops both the pupil mask and the corresponding image to contain just the pupil.
    
    Parameters:
    - pupil: A 2D NumPy array (binary) representing the pupil (1 inside the pupil, 0 outside).
    - image: A 2D NumPy array of the same shape as 'pupil' representing the image to be cropped.
    
    Returns:
    - cropped_pupil: The cropped pupil mask.
    - cropped_image: The cropped image based on the pupil's bounding box.
    """
    # Ensure both arrays have the same shape
    if pupil.shape != image.shape:
        raise ValueError("Pupil and image must have the same dimensions.")

    # Sum along the rows (axis=1) to find the non-zero rows (pupil region)
    row_sums = np.sum(pupil, axis=1)
    non_zero_rows = np.where(row_sums > 0)[0]

    # Sum along the columns (axis=0) to find the non-zero columns (pupil region)
    col_sums = np.sum(pupil, axis=0)
    non_zero_cols = np.where(col_sums > 0)[0]

    # Get the bounding box of the pupil by identifying the min and max indices
    row_start, row_end = non_zero_rows[0], non_zero_rows[-1] + 1
    col_start, col_end = non_zero_cols[0], non_zero_cols[-1] + 1

    # Crop both the pupil and the image
    cropped_pupil = pupil[row_start:row_end, col_start:col_end]
    cropped_image = image[row_start:row_end, col_start:col_end]

    return cropped_pupil, cropped_image






def create_phase_screen_cmd_for_DM(scrn,  scaling_factor=0.1, drop_indicies = None, plot_cmd=False):
    """
    aggregate a scrn (aotools.infinitephasescreen object) onto a DM command space. phase screen is normalized by
    between +-0.5 and then scaled by scaling_factor. Final DM command values should
    always be between -0.5,0.5 (this should be added to a flat reference so flat reference + phase screen should always be bounded between 0-1). phase screens are usually a NxN matrix, while DM is MxM with some missing pixels (e.g. 
    corners). drop_indicies is a list of indicies in the flat MxM DM array that should not be included in the command space. 
    """

    #print('----------\ncheck phase screen size is multiple of DM\n--------')
    
    Nx_act = 12 #number of actuators across DM diameter
    
    scrn_array = ( scrn.scrn - np.min(scrn.scrn) ) / (np.max(scrn.scrn) - np.min(scrn.scrn)) - 0.5 # normalize phase screen between -0.5 - 0.5 
    
    size_factor = int(scrn_array.shape[0] / Nx_act) # how much bigger phase screen is to DM shape in x axis. Note this should be an integer!!
    
    # reshape screen so that axis 1,3 correspond to values that should be aggregated 
    scrn_to_aggregate = scrn_array.reshape(scrn_array.shape[0]//size_factor, size_factor, scrn_array.shape[1]//size_factor, size_factor)
    
    # now aggreagate and apply the scaling factor 
    scrn_on_DM = scaling_factor * np.mean( scrn_to_aggregate, axis=(1,3) ).reshape(-1) 

    #If DM is missing corners etc we set these to nan and drop them before sending the DM command vector
    #dm_cmd =  scrn_on_DM.to_list()
    if drop_indicies is not None:
        for i in drop_indicies:
            scrn_on_DM[i]=np.nan
             
    if plot_cmd: #can be used as a check that the command looks right!
        fig,ax = plt.subplots(1,2,figsize=(12,6))
        im0 = ax[0].imshow( scrn_on_DM.reshape([Nx_act,Nx_act]) )
        ax[0].set_title('DM command (averaging offset)')
        im1 = ax[1].imshow(scrn.scrn)
        ax[1].set_title('original phase screen')
        plt.colorbar(im0, ax=ax[0])
        plt.colorbar(im1, ax=ax[1]) 
        plt.show() 

    dm_cmd =  list( scrn_on_DM[np.isfinite(scrn_on_DM)] ) #drop non-finite values which should be nan values created from drop_indicies array
    return(dm_cmd) 





def magnitude_to_photon_flux(magnitude, band, wavelength):
    """
    Convert stellar magnitude in a given band to photon flux (photons / s / m^2 / nm).
    
    ***EXPERIMENTAL  - need to verify results 
    
    Parameters:
    - magnitude: The magnitude of the star.
    - band: The name of the filter (e.g., 'V', 'J', 'H').
    - wavelength: The central wavelength of the filter in nm.
    
    Returns:
    - photon_flux: The number of photons per second per square meter per nanometer.
    """

    from astropy.constants import h, c
    # Zero points in energy flux for different bands (in erg/s/cm^2/Å)
    zero_point_flux = {
        'V': 3.63e-9 * u.erg / (u.cm**2 * u.s * u.AA),  # V-band zero point
        'J': 3.13e-10 * u.erg / (u.cm**2 * u.s * u.AA), # J-band zero point
        'H': 1.16e-10 * u.erg / (u.cm**2 * u.s * u.AA), # H-band zero point
        # Add more bands as needed
    }
    
    if band not in zero_point_flux:
        raise ValueError(f"Unknown band: {band}. Available bands are {list(zero_point_flux.keys())}")
    
    # Convert magnitude to energy flux density (f_lambda in erg/s/cm^2/Å)
    f_lambda = zero_point_flux[band] * 10**(-0.4 * magnitude)
    
    # Convert wavelength to meters
    wavelength_m = (wavelength * u.nm).to(u.m)
    
    # Convert energy flux density to W/m^2/nm
    f_lambda_si = f_lambda.to(u.W / (u.m**2 * u.nm), equivalencies=u.spectral_density(wavelength_m))
    
    # Calculate the energy per photon (in joules) at the given wavelength
    energy_per_photon = (h * c / wavelength_m).to(u.J)  # Energy per photon at this wavelength
    
    # Calculate photon flux (photons/s/m^2/nm)
    photon_flux = f_lambda_si / energy_per_photon.value  # Explicitly divide by the scalar value of energy_per_photon
    
    # Return photon flux in the appropriate units (photon/s/m^2/nm)
    return photon_flux.value





###### MODELLING MIRROR SCRATCHES


def apply_parabolic_scratches(array, dx, dy, list_a, list_b, list_c, width_list, depth_list):
    """
    Apply multiple parabolic scratches to a 2D array based on input parameters.

    Parameters:
    array (2D numpy array): The input 2D array to which the scratches will be applied.
    dx (float): Pixel scale in the x direction.
    dy (float): Pixel scale in the y direction.
    list_a, list_b, list_c (lists of floats): Lists of a, b, c coefficients for each parabola (y = a*x^2 + b*x + c).
    width_list (list of floats): List of widths for each scratch around the parabolic contour.
    depth_list (list of floats): List of depths for each scratch.

    Returns:
    Modified 2D numpy array with the parabolic scratches applied.
    """
    num_pixels_y, num_pixels_x = array.shape

    # Generate x and y coordinates corresponding to pixel locations
    x_vals = np.linspace(-num_pixels_x/2 * dx, num_pixels_x/2 * dx, num_pixels_x)
    y_vals = np.linspace(-num_pixels_y/2 * dy, num_pixels_y/2 * dy, num_pixels_y)
    X, Y = np.meshgrid(x_vals, y_vals)
    
    # Apply each parabolic scratch
    for a, b, c, width, depth in zip(list_a, list_b, list_c, width_list, depth_list):
        # Compute the parabolic contour y = a*x^2 + b*x + c for each scratch
        parabolic_curve_y = a * X**2 + b * X + c
        
        # Apply the scratch around the parabolic curve with constant depth and width
        for i in range(num_pixels_y):
            for j in range(num_pixels_x):
                # Compute the distance to the parabolic contour
                distance_to_parabola = np.abs(Y[i, j] - parabolic_curve_y[i, j])
                
                # If the point is within the width of the scratch, modify the array
                if distance_to_parabola <= width / 2:
                    array[i, j] -= depth

    return array

# # Example usage
# num_pixels_x, num_pixels_y = 100, 100  # Size of the array (100x100)
# dx, dy = 0.1, 0.1  # Pixel scale in the x and y directions
# input_array = np.full((num_pixels_y, num_pixels_x), 10)  # Constant background array

# # Lists of parabolic parameters, widths, and depths for the scratches
# list_a = [0.5, 0.7]
# list_b = [0, 0]
# list_c = [0, 2]
# width_list = [0.5, 0.2]  # Width of the scratches
# depth_list = [2, 3]  # Depth of the scratches

# # Apply the scratches
# modified_array = apply_parabolic_scratches(input_array, dx, dy, list_a, list_b, list_c, width_list, depth_list)

# # Visualize the result
# import matplotlib.pyplot as plt
# plt.imshow(modified_array, cmap='hot', extent=[-num_pixels_x/2*dx, num_pixels_x/2*dx, -num_pixels_y/2*dy, num_pixels_y/2*dy])
# plt.colorbar(label='Value')
# plt.title('2D Array with Multiple Parabolic Scratches')
# plt.show()





#### PLOTTING 


def nice_heatmap_subplots( im_list , xlabel_list=None, ylabel_list=None, title_list=None, cbar_label_list=None, fontsize=15, cbar_orientation = 'bottom', axis_off=True, vlims=None, savefig=None):

    n = len(im_list)
    fs = fontsize
    fig = plt.figure(figsize=(5*n, 5))

    for a in range(n) :
        ax1 = fig.add_subplot(int(f'1{n}{a+1}'))

        if vlims is not None:
            im1 = ax1.imshow(  im_list[a] , vmin = vlims[a][0], vmax = vlims[a][1])
        else:
            im1 = ax1.imshow(  im_list[a] )
        if title_list is not None:
            ax1.set_title( title_list[a] ,fontsize=fs)
        if xlabel_list is not None:
            ax1.set_xlabel( xlabel_list[a] ,fontsize=fs) 
        if ylabel_list is not None:
            ax1.set_ylabel( ylabel_list[a] ,fontsize=fs) 
        ax1.tick_params( labelsize=fs ) 

        if axis_off:
            ax1.axis('off')
        divider = make_axes_locatable(ax1)
        if cbar_orientation == 'bottom':
            cax = divider.append_axes('bottom', size='5%', pad=0.05)
            cbar = fig.colorbar( im1, cax=cax, orientation='horizontal')
                
        elif cbar_orientation == 'top':
            cax = divider.append_axes('top', size='5%', pad=0.05)
            cbar = fig.colorbar( im1, cax=cax, orientation='horizontal')
                
        else: # we put it on the right 
            cax = divider.append_axes('right', size='5%', pad=0.05)
            cbar = fig.colorbar( im1, cax=cax, orientation='vertical')  
        
        if cbar_label_list is not None:
            cbar.set_label( cbar_label_list[a], rotation=0,fontsize=fs)
        cbar.ax.tick_params(labelsize=fs)
    if savefig is not None:
        plt.savefig( savefig , bbox_inches='tight', dpi=300) 

    #plt.show()
    
    
     

def nice_DM_plot( data, savefig=None ): #for a 140 actuator BMC 3.5 DM
    fig,ax = plt.subplots(1,1)
    if len( np.array(data).shape ) == 1: 
        ax.imshow( get_DM_command_in_2D(data) )
    else: 
        ax.imshow( data )
    #ax.set_title('poorly registered actuators')
    ax.grid(True, which='minor',axis='both', linestyle='-', color='k', lw=2 )
    ax.set_xticks( np.arange(12) - 0.5 , minor=True)
    ax.set_yticks( np.arange(12) - 0.5 , minor=True)
    if savefig is not None:
        plt.savefig( savefig , bbox_inches='tight', dpi=300) 



def plot_data_and_residuals(x, y_meas, y_model, xlabel, ylabel, residual_ylabel, label_1=None, label_2=None, savefig=None):
    # Calculate residuals
    residuals = y_meas - y_model

    # Create a figure with two subplots: one for the data and one for the residuals
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 6), gridspec_kw={'height_ratios': [3, 1]}, sharex=True)
    
    # First subplot: measured and modeled data
    if label_1 is None:
        ax1.plot(x, y_meas, '-', label='Measured Data', color='blue', markersize=2, alpha =0.3)
    else: 
        ax1.plot(x, y_meas, '-', label=label_1, color='blue', markersize=2, alpha =0.3)
    if label_2 is None:    
        ax1.plot(x, y_model, '.', label='Modeled Data', color='red', linewidth=2, alpha =0.3)
    else:
        ax1.plot(x, y_model, '.', label=label_2, color='red', linewidth=2, alpha =0.3)
    #ax1.set_xlabel(xlabel)
    ax1.set_ylabel(ylabel)
    ax1.legend()
    ax1.grid(True)

    # Second subplot: residuals
    ax2.plot(x, residuals, '.', color='green', markersize=5, alpha =0.3)
    ax2.axhline(0, color='black', linewidth=1, linestyle='--')  # Horizontal line at zero
    ax2.set_xlabel(xlabel)
    ax2.set_ylabel(residual_ylabel)
    ax2.grid(True)

    # Adjust layout for better spacing between subplots
    plt.tight_layout()
    
    if savefig is not None:
        plt.savefig(savefig, bbox_inches='tight', dpi=300  )
    # Show the plot
    plt.show()
    
    

def create_telem_mosaic(image_list, image_title_list, image_colorbar_list, 
                  plot_list, plot_title_list, plot_xlabel_list, plot_ylabel_list):
    """
    Creates a 3-row mosaic layout with:
    - First row: images with colorbars below
    - Second and third rows: plots with titles and axis labels
    
    Parameters:
    - image_list: List of image data for the first row (4 images)
    - image_title_list: List of titles for the first row images
    - image_colorbar_list: List of colorbars (True/False) for each image in the first row
    - plot_list: List of plot data for second and third rows (4 plots, 2 per row)
    - plot_title_list: List of titles for each plot
    - plot_xlabel_list: List of x-axis labels for each plot
    - plot_ylabel_list: List of y-axis labels for each plot
    """
    
    # Create a figure with constrained layout and extra padding
    fig = plt.figure(constrained_layout=True, figsize=(10, 8))
    plt.subplots_adjust(wspace=0.4, hspace=0.4)
    
    # Create GridSpec with 3 rows and different numbers of columns
    gs = GridSpec(3, 4, figure=fig, height_ratios=[1, 1, 1])
    
    # Top row: 4 columns with colorbars
    for i in range(4):
        ax = fig.add_subplot(gs[0, i])
        img = image_list[i]
        im = ax.imshow(img, cmap='viridis')  # Modify colormap if needed
        ax.set_title(image_title_list[i])
        
        # Optionally add a colorbar below the image
        if image_colorbar_list[i]:
            divider = make_axes_locatable(ax)
            cax = divider.append_axes('bottom', size='5%', pad=0.2)
            fig.colorbar(im, cax=cax, orientation='horizontal')
    
    # Middle row: 2 columns, each spanning 2 grid columns
    for i in range(2):
        ax = fig.add_subplot(gs[1, 2*i:2*i+2])
        data = plot_list[i]
        ax.plot(data)
        ax.set_title(plot_title_list[i])
        ax.set_xlabel(plot_xlabel_list[i])
        ax.set_ylabel(plot_ylabel_list[i])

    # Bottom row: 2 columns, each spanning 2 grid columns
    for i in range(2, 4):
        ax = fig.add_subplot(gs[2, 2*(i-2):2*(i-2)+2])
        data = plot_list[i]
        
        ax.plot(data)
        ax.set_title(plot_title_list[i])
        ax.set_xlabel(plot_xlabel_list[i])
        ax.set_ylabel(plot_ylabel_list[i])
    
    # Show the plot
    plt.show()



def plot_eigenmodes( zwfs_ns , save_path = None ):
    
    tstamp = datetime.datetime.now().strftime("%d-%m-%YT%H.%M.%S")

    U,S,Vt = np.linalg.svd( zwfs_ns.reco.IM, full_matrices=True)

    #singular values
    plt.figure() 
    plt.semilogy(S) #/np.max(S))
    #plt.axvline( np.pi * (10/2)**2, color='k', ls=':', label='number of actuators in pupil')
    plt.legend() 
    plt.xlabel('mode index')
    plt.ylabel('singular values')

    if save_path is not None:
        plt.savefig(save_path +  f'singularvalues_{tstamp}.png', bbox_inches='tight', dpi=200)
    plt.show()
    
    # THE IMAGE MODES 
    n_row = round( np.sqrt( zwfs_ns.reco.M2C_0.shape[0]) ) - 1
    fig,ax = plt.subplots(n_row  ,n_row ,figsize=(30,30))
    plt.subplots_adjust(hspace=0.1,wspace=0.1)
    for i,axx in enumerate(ax.reshape(-1)):
        # we filtered circle on grid, so need to put back in grid
        tmp =  zwfs_ns.pupil_regions.pupil_filt.copy()
        vtgrid = np.zeros(tmp.shape)
        vtgrid[tmp] = Vt[i]
        r1,r2,c1,c2 = 10,-10,10,-10
        axx.imshow( vtgrid.reshape(zwfs_ns.reco.I0.shape )[r1:r2,c1:c2] ) #cp_x2-cp_x1,cp_y2-cp_y1) )
        #axx.set_title(f'\n\n\nmode {i}, S={round(S[i]/np.max(S),3)}',fontsize=5)
        #
        axx.text( 10,10, f'{i}',color='w',fontsize=4)
        axx.text( 10,20, f'S={round(S[i]/np.max(S),3)}',color='w',fontsize=4)
        axx.axis('off')
        #plt.legend(ax=axx)
    plt.tight_layout()

    if save_path is not None:
        plt.savefig(save_path + f'det_eignmodes_{tstamp}.png',bbox_inches='tight',dpi=200)
    plt.show()
    
    # THE DM MODES 

    # NOTE: if not zonal (modal) i might need M2C to get this to dm space 
    # if zonal M2C is just identity matrix. 
    fig,ax = plt.subplots(n_row, n_row, figsize=(30,30))
    plt.subplots_adjust(hspace=0.1,wspace=0.1)
    for i,axx in enumerate(ax.reshape(-1)):
        axx.imshow( get_DM_command_in_2D( zwfs_ns.reco.M2C_0.T @ U.T[i] ) )
        #axx.set_title(f'mode {i}, S={round(S[i]/np.max(S),3)}')
        axx.text( 1,2,f'{i}',color='w',fontsize=6)
        axx.text( 1,3,f'S={round(S[i]/np.max(S),3)}',color='w',fontsize=6)
        axx.axis('off')
        #plt.legend(ax=axx)
    plt.tight_layout()
    if save_path is not None:
        plt.savefig(save_path +  f'dm_eignmodes_{tstamp}.png',bbox_inches='tight',dpi=200)
    plt.show()




def display_images_with_slider(image_lists, plot_titles=None, cbar_labels=None):
    """
    Displays multiple images or 1D plots from a list of lists with a slider to control the shared index.
    
    Parameters:
    - image_lists: list of lists where each inner list contains either 2D arrays (images) or 1D arrays (scalars).
                   The inner lists must all have the same length.
    - plot_titles: list of strings, one for each subplot. Default is None (no titles).
    - cbar_labels: list of strings, one for each colorbar. Default is None (no labels).
    """
    
    # Check that all inner lists have the same length
    assert all(len(lst) == len(image_lists[0]) for lst in image_lists), "All inner lists must have the same length."
    
    # Number of rows and columns based on the number of plots
    num_plots = len(image_lists)
    ncols = math.ceil(math.sqrt(num_plots))  # Number of columns for grid
    nrows = math.ceil(num_plots / ncols)     # Number of rows for grid
    
    num_frames = len(image_lists[0])

    # Create figure and axes
    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(5 * ncols, 5 * nrows))
    plt.subplots_adjust(bottom=0.2)

    # Flatten axes array for easier iteration
    axes = axes.flatten() if num_plots > 1 else [axes]

    # Store the display objects for each plot (either imshow or line plot)
    img_displays = []
    line_displays = []
    
    # Get max/min values for 1D arrays to set static axis limits
    max_values = [max(lst) if not isinstance(lst[0], np.ndarray) else None for lst in image_lists]
    min_values = [min(lst) if not isinstance(lst[0], np.ndarray) else None for lst in image_lists]

    for i, ax in enumerate(axes[:num_plots]):  # Only iterate over the number of plots
        # Check if the first item in the list is a 2D array (an image) or a scalar
        if isinstance(image_lists[i][0], np.ndarray) and image_lists[i][0].ndim == 2:
            # Use imshow for 2D data (images)
            img_display = ax.imshow(image_lists[i][0], cmap='viridis')
            img_displays.append(img_display)
            line_displays.append(None)  # Placeholder for line plots
            
            # Add colorbar if it's an image
            cbar = fig.colorbar(img_display, ax=ax)
            if cbar_labels and i < len(cbar_labels) and cbar_labels[i] is not None:
                cbar.set_label(cbar_labels[i])

        else:
            # Plot the list of scalar values up to the initial index
            line_display, = ax.plot(np.arange(len(image_lists[i])), image_lists[i], color='b')
            line_display.set_data(np.arange(1), image_lists[i][:1])  # Start with only the first value
            ax.set_xlim(0, len(image_lists[i]))  # Set x-axis to full length of the data
            ax.set_ylim(min_values[i], max_values[i])  # Set y-axis to cover the full range
            line_displays.append(line_display)
            img_displays.append(None)  # Placeholder for image plots

        # Set plot title if provided
        if plot_titles and i < len(plot_titles) and plot_titles[i] is not None:
            ax.set_title(plot_titles[i])

    # Remove any unused axes
    for ax in axes[num_plots:]:
        ax.remove()

    # Slider for selecting the frame index
    ax_slider = plt.axes([0.2, 0.05, 0.65, 0.03], facecolor='lightgoldenrodyellow')
    frame_slider = Slider(ax_slider, 'Frame', 0, num_frames - 1, valinit=0, valstep=1)

    # Update function for the slider
    def update(val):
        index = int(frame_slider.val)  # Get the selected index from the slider
        for i, (img_display, line_display) in enumerate(zip(img_displays, line_displays)):
            if img_display is not None:
                # Update the image data for 2D data
                img_display.set_data(image_lists[i][index])
            if line_display is not None:
                # Update the line plot for scalar values (plot up to the selected index)
                line_display.set_data(np.arange(index), image_lists[i][:index])
        fig.canvas.draw_idle()  # Redraw the figure

    # Connect the slider to the update function
    frame_slider.on_changed(update)

    plt.show()



def display_images_as_movie(image_lists, plot_titles=None, cbar_labels=None, save_path="output_movie.mp4", fps=5):
    """
    Creates an animation from multiple images or 1D plots from a list of lists and saves it as a movie.
    
    Parameters:
    - image_lists: list of lists where each inner list contains either 2D arrays (images) or 1D arrays (scalars).
                   The inner lists must all have the same length.
    - plot_titles: list of strings, one for each subplot. Default is None (no titles).
    - cbar_labels: list of strings, one for each colorbar. Default is None (no labels).
    - save_path: path where the output movie will be saved.
    - fps: frames per second for the output movie.
    """
    
    # Check that all inner lists have the same length
    assert all(len(lst) == len(image_lists[0]) for lst in image_lists), "All inner lists must have the same length."
    
    # Number of rows and columns based on the number of plots
    num_plots = len(image_lists)
    ncols = math.ceil(math.sqrt(num_plots))  # Number of columns for grid
    nrows = math.ceil(num_plots / ncols)     # Number of rows for grid
    
    num_frames = len(image_lists[0])

    # Create figure and axes
    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(6 * ncols, 6 * nrows))
    plt.subplots_adjust(bottom=0.2)

    # Flatten axes array for easier iteration
    axes = axes.flatten() if num_plots > 1 else [axes]

    # Store the display objects for each plot (either imshow or line plot)
    img_displays = []
    line_displays = []
    
    # Get max/min values for 1D arrays to set static axis limits
    max_values = [max(lst) if not isinstance(lst[0], np.ndarray) else None for lst in image_lists]
    min_values = [min(lst) if not isinstance(lst[0], np.ndarray) else None for lst in image_lists]

    for i, ax in enumerate(axes[:num_plots]):  # Only iterate over the number of plots
        # Check if the first item in the list is a 2D array (an image) or a scalar
        if isinstance(image_lists[i][0], np.ndarray) and image_lists[i][0].ndim == 2:
            # Use imshow for 2D data (images)
            img_display = ax.imshow(image_lists[i][0], cmap='viridis')
            img_displays.append(img_display)
            line_displays.append(None)  # Placeholder for line plots
            
            # Add colorbar if it's an image
            cbar = fig.colorbar(img_display, ax=ax)
            if cbar_labels and i < len(cbar_labels) and cbar_labels[i] is not None:
                cbar.set_label(cbar_labels[i])

        else:
            # Plot the list of scalar values up to the initial index
            line_display, = ax.plot(np.arange(len(image_lists[i])), image_lists[i], color='b')
            line_display.set_data(np.arange(1), image_lists[i][:1])  # Start with only the first value
            ax.set_xlim(0, len(image_lists[i]))  # Set x-axis to full length of the data
            ax.set_ylim(min_values[i], max_values[i])  # Set y-axis to cover the full range
            line_displays.append(line_display)
            img_displays.append(None)  # Placeholder for image plots

        # Set plot title if provided
        if plot_titles and i < len(plot_titles) and plot_titles[i] is not None:
            ax.set_title(plot_titles[i])

    # Remove any unused axes
    for ax in axes[num_plots:]:
        ax.remove()

    # Function to update the frames
    def update_frame(frame_idx):
        for i, (img_display, line_display) in enumerate(zip(img_displays, line_displays)):
            if img_display is not None:
                # Update the image data for 2D data
                img_display.set_data(image_lists[i][frame_idx])
            if line_display is not None:
                # Update the line plot for scalar values (plot up to the current index)
                line_display.set_data(np.arange(frame_idx), image_lists[i][:frame_idx])
        return img_displays + line_displays

    # Create the animation
    ani = animation.FuncAnimation(fig, update_frame, frames=num_frames, blit=False, repeat=False)

    # Save the animation as a movie file
    ani.save(save_path, fps=fps, writer='ffmpeg')

    plt.show()

