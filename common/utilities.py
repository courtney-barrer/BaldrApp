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
