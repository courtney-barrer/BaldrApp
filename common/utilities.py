import numpy as np 
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from mpl_toolkits.axes_grid1 import make_axes_locatable
import datetime 

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




def nice_heatmap_subplots( im_list , xlabel_list, ylabel_list, title_list,cbar_label_list, fontsize=15, cbar_orientation = 'bottom', axis_off=True, vlims=None, savefig=None):

    n = len(im_list)
    fs = fontsize
    fig = plt.figure(figsize=(5*n, 5))

    for a in range(n) :
        ax1 = fig.add_subplot(int(f'1{n}{a+1}'))
        ax1.set_title(title_list[a] ,fontsize=fs)

        if vlims is not None:
            im1 = ax1.imshow(  im_list[a] , vmin = vlims[a][0], vmax = vlims[a][1])
        else:
            im1 = ax1.imshow(  im_list[a] )
        ax1.set_title( title_list[a] ,fontsize=fs)
        ax1.set_xlabel( xlabel_list[a] ,fontsize=fs) 
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
    ax.set_title('poorly registered actuators')
    ax.grid(True, which='minor',axis='both', linestyle='-', color='k', lw=2 )
    ax.set_xticks( np.arange(12) - 0.5 , minor=True)
    ax.set_yticks( np.arange(12) - 0.5 , minor=True)
    if savefig is not None:
        plt.savefig( savefig , bbox_inches='tight', dpi=300) 



# Define Cauchy's equation
def cauchy_eqn(wavelength, A, B, C):
    return A + B / wavelength**2 + C / wavelength**4


def fit_cauchy_eqn_to_data(df, savefig = None):
    # Example dataframe (replace with your actual df)
    # df = pd.read_csv('path_to_your_data.csv')  # If your data is stored in a CSV
    # Assuming your data is already in a dataframe 'df'
    wavelengths = df['Wavelength(nm)'].values  # Extracting wavelength values
    n_measured = df['n'].values  # Extracting refractive index values

    # Perform the curve fitting
    popt, pcov = curve_fit(cauchy_eqn, wavelengths, n_measured)

    # Extract the fitted coefficients
    A_fit, B_fit, C_fit = popt

    # Generate the fitted curve
    n_fitted = cauchy_eqn(wavelengths, A_fit, B_fit, C_fit)

    # Plot the measured data vs the fitted curve
    plt.plot(wavelengths, n_measured, 'b-', label='Measured Data')
    plt.plot(wavelengths, n_fitted, 'r--', label=f'Fitted Cauchy Eqn\nA={A_fit:.4f}, B={B_fit:.4e}, C={C_fit:.4e}')
    plt.xlabel('Wavelength (nm)')
    plt.ylabel('Refractive Index n')
    plt.legend()
    if savefig is not None:
        plt.savefig(savefig, bbox_inches='tight', dpi=200)  
    plt.show()
