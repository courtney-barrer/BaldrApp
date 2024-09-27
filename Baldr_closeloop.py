import numpy as np
import matplotlib.pyplot as plt

def get_theoretical_reference_pupils( wavelength = 1.65e-6 ,F_number = 21.2, mask_diam = 1.2, diameter_in_angular_units = True, get_individual_terms=False, phaseshift = np.pi/2 , padding_factor = 4, debug= True, analytic_solution = True ) :
    """
    get theoretical reference pupil intensities of ZWFS with / without phasemask 
    NO ABERRATIONS

    Parameters
    ----------
    wavelength : TYPE, optional
        DESCRIPTION. input wavelength The default is 1.65e-6.
    F_number : TYPE, optional
        DESCRIPTION. The default is 21.2.
    mask_diam : phase dot diameter. TYPE, optional
            if diameter_in_angular_units=True than this has diffraction limit units ( 1.22 * f * lambda/D )
            if  diameter_in_angular_units=False than this has physical units (m) determined by F_number and wavelength
        DESCRIPTION. The default is 1.2.
    diameter_in_angular_units : TYPE, optional
        DESCRIPTION. The default is True.
    get_individual_terms : Type optional
        DESCRIPTION : if false (default) with jsut return intensity, otherwise return P^2, abs(M)^2 , phi + mu
    phaseshift : TYPE, optional
        DESCRIPTION. phase phase shift imparted on input field (radians). The default is np.pi/2.
    padding_factor : pad to change the resolution in image plane. TYPE, optional
        DESCRIPTION. The default is 4.
    debug : TYPE, optional
        DESCRIPTION. Do we want to plot some things? The default is True.
    analytic_solution: TYPE, optional
        DESCRIPTION. use analytic formula or calculate numerically? The default is True.
    Returns
    -------
    Ic, reference pupil intensity with phasemask in 
    P, reference pupil intensity with phasemask out 

    """
    pupil_radius = 1  # Pupil radius in meters

    # Define the grid in the pupil plane
    N = 2**9 + 1 #256  # Number of grid points (assumed to be square)
    L_pupil = 2 * pupil_radius  # Pupil plane size (physical dimension)
    dx_pupil = L_pupil / N  # Sampling interval in the pupil plane
    x_pupil = np.linspace(-L_pupil/2, L_pupil/2, N)   # Pupil plane coordinates
    y_pupil = np.linspace(-L_pupil/2, L_pupil/2, N) 
    X_pupil, Y_pupil = np.meshgrid(x_pupil, y_pupil)
    
    


    # Define a circular pupil function
    pupil = np.sqrt(X_pupil**2 + Y_pupil**2) <= pupil_radius

    # Zero padding to increase resolution
    # Increase the array size by padding (e.g., 4x original size)
    N_padded = N * padding_factor
    pupil_padded = np.zeros((N_padded, N_padded))
    start_idx = (N_padded - N) // 2
    pupil_padded[start_idx:start_idx+N, start_idx:start_idx+N] = pupil

    # Perform the Fourier transform on the padded array (normalizing for the FFT)
    pupil_ft = np.fft.fftshift(np.fft.fft2(np.fft.ifftshift(pupil_padded)))
    
    # Compute the Airy disk scaling factor (1.22 * λ * F)
    airy_scale = 1.22 * wavelength * F_number

    # Image plane sampling interval (adjusted for padding)
    L_image = wavelength * F_number / dx_pupil  # Total size in the image plane
    dx_image_padded = L_image / N_padded  # Sampling interval in the image plane with padding
    
    if diameter_in_angular_units:
        x_image_padded = np.linspace(-L_image/2, L_image/2, N_padded) / airy_scale  # Image plane coordinates in Airy units
        y_image_padded = np.linspace(-L_image/2, L_image/2, N_padded) / airy_scale
    else:
        x_image_padded = np.linspace(-L_image/2, L_image/2, N_padded)  # Image plane coordinates in Airy units
        y_image_padded = np.linspace(-L_image/2, L_image/2, N_padded) 
        
    X_image_padded, Y_image_padded = np.meshgrid(x_image_padded, y_image_padded)

    if diameter_in_angular_units:
        mask = np.sqrt(X_image_padded**2 + Y_image_padded**2) <= mask_diam / 4
    else: 
        mask = np.sqrt(X_image_padded**2 + Y_image_padded**2) <= mask_diam / 4
        
    
    psi_B = np.fft.fftshift(np.fft.fft2(np.fft.ifftshift(pupil_padded)) )
                            
    b = np.fft.fftshift( np.fft.ifft2( mask * psi_B ) ) 

    
    if debug: 
        
        psf = np.abs(pupil_ft)**2  # Get the PSF by taking the square of the absolute value
        psf /= np.max(psf)  # Normalize PSF intensity
        
        if diameter_in_angular_units:
            zoom_range = 3  # Number of Airy disk radii to zoom in on
        else:
            zoom_range = 3 * airy_scale 
            
        extent = (-zoom_range, zoom_range, -zoom_range, zoom_range)

        fig,ax = plt.subplots(1,1)
        ax.imshow(psf, extent=(x_image_padded.min(), x_image_padded.max(), y_image_padded.min(), y_image_padded.max()), cmap='gray')
        ax.contour(X_image_padded, Y_image_padded, mask, levels=[0.5], colors='red', linewidths=2, label='phasemask')
        #ax[1].imshow( mask, extent=(x_image_padded.min(), x_image_padded.max(), y_image_padded.min(), y_image_padded.max()), cmap='gray')
        #for axx in ax.reshape(-1):
        #    axx.set_xlim(-zoom_range, zoom_range)
        #    axx.set_ylim(-zoom_range, zoom_range)
        ax.set_xlim(-zoom_range, zoom_range)
        ax.set_ylim(-zoom_range, zoom_range)
        ax.set_title( 'PSF' )
        ax.legend() 
        #ax[1].set_title('phasemask')


    
    # if considering complex b 
    # beta = np.angle(b) # complex argunment of b 
    # M = b * (np.exp(1J*theta)-1)**0.5
    
    # relabelling
    theta = phaseshift # rad , 
    P = pupil_padded.copy() 
    
    if analytic_solution :
        
        M = abs( b ) * np.sqrt((np.cos(theta)-1)**2 + np.sin(theta)**2)
        mu = np.angle((np.exp(1J*theta)-1) ) # np.arctan( np.sin(theta)/(np.cos(theta)-1) ) #
        
        phi = np.zeros( P.shape ) # added aberrations 
        
        # out formula ----------
        #if measured_pupil!=None:
        #    P = measured_pupil / np.mean( P[P > np.mean(P)] ) # normalize by average value in Pupil
        
        Ic = ( P**2 + abs(M)**2 + 2* P* abs(M) * np.cos(phi + mu) ) #+ beta)
        if not get_individual_terms:
            return( P, Ic )
        else:
            return( P, abs(M) , phi+mu )
    else:
        
        # phasemask filter 
        
        T_on = 1
        T_off = 1
        H = T_off*(1 + (T_on/T_off * np.exp(1j * theta) - 1) * mask  ) 
        
        Ic = abs( np.fft.fftshift( np.fft.ifft2( H * psi_B ) ) ) **2 
    
        return( P, Ic)





def get_grids( wavelength = 1.65e-6 , F_number = 21.2, mask_diam = 1.2, diameter_in_angular_units = True, N = 256, padding_factor = 4 ) :
    """
    get theoretical reference pupil intensities of ZWFS with / without phasemask 
    

    Parameters
    ----------
    wavelength : TYPE, optional
        DESCRIPTION. input wavelength The default is 1.65e-6.
    F_number : TYPE, optional
        DESCRIPTION. The default is 21.2.
    mask_diam : phase dot diameter. TYPE, optional
            if diameter_in_angular_units=True than this has diffraction limit units ( 1.22 * f * lambda/D )
            if  diameter_in_angular_units=False than this has physical units (m) determined by F_number and wavelength
        DESCRIPTION. The default is 1.2.
    diameter_in_angular_units : TYPE, optional
        DESCRIPTION. The default is True.
    get_individual_terms : Type optional
        DESCRIPTION : if false (default) with jsut return intensity, otherwise return P^2, abs(M)^2 , phi + mu
    phaseshift : TYPE, optional
        DESCRIPTION. phase phase shift imparted on input field (radians). The default is np.pi/2.
    padding_factor : pad to change the resolution in image plane. TYPE, optional
        DESCRIPTION. The default is 4.
    debug : TYPE, optional
        DESCRIPTION. Do we want to plot some things? The default is True.
    analytic_solution: TYPE, optional
        DESCRIPTION. use analytic formula or calculate numerically? The default is True.
    Returns
    -------
    Ic, reference pupil intensity with phasemask in 
    P, reference pupil intensity with phasemask out 

    """
    pupil_radius = 1  # Pupil radius in meters

    # Define the grid in the pupil plane
    #N = 2**9 + 1 #256  # Number of grid points (assumed to be square)
    L_pupil = 2 * pupil_radius  # Pupil plane size (physical dimension)
    dx_pupil = L_pupil / N  # Sampling interval in the pupil plane
    x_pupil = np.linspace(-L_pupil/2, L_pupil/2, N)   # Pupil plane coordinates
    y_pupil = np.linspace(-L_pupil/2, L_pupil/2, N) 
    X_pupil, Y_pupil = np.meshgrid(x_pupil, y_pupil)
    

    # Define a circular pupil function
    pupil = np.sqrt(X_pupil**2 + Y_pupil**2) <= pupil_radius

    # Zero padding to increase resolution
    # Increase the array size by padding (e.g., 4x original size)
    N_padded = N * padding_factor
    pupil_padded = np.zeros((N_padded, N_padded))
    start_idx = (N_padded - N) // 2
    pupil_padded[start_idx:start_idx+N, start_idx:start_idx+N] = pupil

    # Perform the Fourier transform on the padded array (normalizing for the FFT)
    #pupil_ft = np.fft.fftshift(np.fft.fft2(np.fft.ifftshift(pupil_padded)))
    
    # Compute the Airy disk scaling factor (1.22 * λ * F)
    airy_scale = 1.22 * wavelength * F_number

    # Image plane sampling interval (adjusted for padding)
    L_image = wavelength * F_number / dx_pupil  # Total size in the image plane
    #dx_image_padded = L_image / N_padded  # Sampling interval in the image plane with padding
    
    if diameter_in_angular_units:
        x_image_padded = np.linspace(-L_image/2, L_image/2, N_padded) / airy_scale  # Image plane coordinates in Airy units
        y_image_padded = np.linspace(-L_image/2, L_image/2, N_padded) / airy_scale
    else:
        x_image_padded = np.linspace(-L_image/2, L_image/2, N_padded)  # Image plane coordinates in Airy units
        y_image_padded = np.linspace(-L_image/2, L_image/2, N_padded) 
        
    X_image_padded, Y_image_padded = np.meshgrid(x_image_padded, y_image_padded)
    
    if diameter_in_angular_units:
        mask = np.sqrt(X_image_padded**2 + Y_image_padded**2) <= mask_diam / 4
    else: 
        mask = np.sqrt(X_image_padded**2 + Y_image_padded**2) <= mask_diam / 4
        
    return pupil_padded, mask  




# Define Gaussian function for actuator displacement
def gaussian_displacement(c_i, sigma_i, x, y, x0, y0):
    """Compute Gaussian displacement for a single actuator."""
    return c_i * np.exp(-((x - x0)**2 + (y - y0)**2) / sigma_i**2)




def get_DM_command_in_2D(cmd,Nx_act=12):
    # function so we can easily plot the DM shape (since DM grid is not perfectly square raw cmds can not be plotted in 2D immediately )
    #puts nan values in cmd positions that don't correspond to actuator on a square grid until cmd length is square number (12x12 for BMC multi-2.5 DM) so can be reshaped to 2D array to see what the command looks like on the DM.
    corner_indices = [0, Nx_act-1, Nx_act * (Nx_act-1), Nx_act*Nx_act]
    cmd_in_2D = list(cmd.copy())
    for i in corner_indices:
        cmd_in_2D.insert(i,np.nan)
    return( np.array(cmd_in_2D).reshape(Nx_act,Nx_act) )



def generate_dm_coordinates(Nx=12, Ny=12, spacing=1.0):
    """
    Generates the x, y coordinates of the actuators in a 12x12 grid DM with missing corners.
    
    Args:
        Nx, Ny: Number of actuators in the x and y directions (12x12 grid).
        spacing: The spacing between actuators (default is 1 unit).
    
    Returns:
        - coords: A list of tuples (x, y) representing the coordinates of the actuators.
        - flattened_indices: A dictionary that maps actuator indices (0 to 139) to (x, y) coordinates.
        - coord_to_index: A dictionary mapping (x, y) coordinates to actuator indices.
    """
    coords = []
    coord_to_index = {}
    flattened_indices = {}
    
    center_x = (Nx - 1) / 2  # Center of the grid in x
    center_y = (Ny - 1) / 2  # Center of the grid in y
    
    actuator_index = 0
    for i in range(Ny):
        for j in range(Nx):
            # Skip the missing corners
            if (i == 0 and j == 0) or (i == 0 and j == Nx - 1) or (i == Ny - 1 and j == 0) or (i == Ny - 1 and j == Nx - 1):
                continue

            # Calculate x and y coordinates relative to the center
            x = (j - center_x) * spacing
            y = (i - center_y) * spacing
            
            coords.append((x, y))
            coord_to_index[(x, y)] = actuator_index
            flattened_indices[actuator_index] = (x, y)
            actuator_index += 1

    return coords, flattened_indices, coord_to_index


def get_nearest_actuator(x, y, flattened_indices):
    """
    Finds the nearest actuator index for a given (x, y) coordinate.
    
    Args:
        x, y: The (x, y) coordinates to match to the nearest actuator.
        flattened_indices: A dictionary mapping actuator indices to (x, y) coordinates.
    
    Returns:
        Nearest actuator index.
    """
    distances = {index: np.sqrt((x - coord[0])**2 + (y - coord[1])**2) for index, coord in flattened_indices.items()}
    return min(distances, key=distances.get)


def actuator_to_xy(actuator_index, flattened_indices):
    """
    Given an actuator index, return the corresponding (x, y) coordinates.
    
    Args:
        actuator_index: The actuator number in the flattened 140-length array.
        flattened_indices: A dictionary mapping actuator indices to (x, y) coordinates.
    
    Returns:
        (x, y) coordinates of the actuator.
    """
    return flattened_indices.get(actuator_index)


def fit_affine_transformation_with_center(corners_dm, corners_img, intersection_img):
    """
    Fit an affine transformation from DM space to image space, using the DM center as the origin (0,0).
    
    Args:
        corners_dm: List of (x, y) coordinates of DM corners in DM space (relative to the DM center).
        corners_img: List of (x, y) coordinates of the corresponding points in image space.
        intersection_img: The (x, y) coordinates of the DM center in image space.
    
    Returns:
        - transform_matrix: A 2x3 matrix that transforms DM coordinates to pixel coordinates.
    """
    # Create arrays for the corners
    dm = np.array(corners_dm)
    img = np.array(corners_img)

    # Subtract the DM center (intersection) from the image coordinates to compute translation
    tx, ty = intersection_img
    
    # Now we need to solve for the linear transformation matrix (a, b, c, d)
    # We have the relationship: [x_img, y_img] = A * [x_dm, y_dm] + [tx, ty]
    # where A is the 2x2 matrix with components [a, b; c, d]
    
    # Create the matrix for DM space (without the translation part)
    dm_coords = np.vstack([dm.T, np.ones(len(dm))]).T
    
    # Subtract translation from image coordinates (image coordinates relative to DM center)
    img_coords = img - np.array([tx, ty])

    # Solve the linear system A * dm_coords = img_coords for A (a, b, c, d)
    # Solve the two systems independently for x and y
    A_x = np.linalg.lstsq(dm_coords[:, :2], img_coords[:, 0], rcond=None)[0]
    A_y = np.linalg.lstsq(dm_coords[:, :2], img_coords[:, 1], rcond=None)[0]
    
    # Construct the 2x3 affine transformation matrix
    transform_matrix = np.array([
        [A_x[0], A_x[1], tx],  # [a, b, tx]
        [A_y[0], A_y[1], ty]   # [c, d, ty]
    ])
    
    return transform_matrix

def pixel_to_dm(pixel_coord, transform_matrix):
    """
    Converts pixel coordinates to DM coordinates using the inverse of the affine transformation.
    
    Args:
        pixel_coord: A tuple (x, y) in pixel space.
        transform_matrix: The affine transformation matrix from DM space to pixel space.
    
    Returns:
        Tuple (x_dm, y_dm) in DM coordinates.
    """
    A = transform_matrix[:, :2]  # 2x2 matrix part
    t = transform_matrix[:, 2]   # translation part
    
    # Inverse transformation
    A_inv = np.linalg.inv(A)
    pixel_coord = np.array(pixel_coord)
    dm_coord = np.dot(A_inv, pixel_coord - t)
    return tuple(dm_coord)

def dm_to_pixel(dm_coord, transform_matrix):
    """
    Converts DM coordinates to pixel coordinates using the affine transformation.
    
    Args:
        dm_coord: A tuple (x, y) in DM space.
        transform_matrix: The affine transformation matrix from DM space to pixel space.
    
    Returns:
        Tuple (x_pixel, y_pixel) in pixel coordinates.
    """
    dm_coord = np.array(dm_coord)
    pixel_coord = np.dot(transform_matrix[:, :2], dm_coord) + transform_matrix[:, 2]
    return tuple(pixel_coord)



def convert_to_serializable(obj):
    """
    Recursively converts NumPy arrays and other non-serializable objects to serializable forms.
    Also converts dictionary keys to standard types (str, int, float).
    """
    if isinstance(obj, np.ndarray):
        return obj.tolist()  # Convert NumPy arrays to lists
    elif isinstance(obj, np.integer):
        return int(obj)  # Convert NumPy integers to Python int
    elif isinstance(obj, np.floating):
        return float(obj)  # Convert NumPy floats to Python float
    elif isinstance(obj, dict):
        return {str(key): convert_to_serializable(value) for key, value in obj.items()}  # Ensure keys are strings
    elif isinstance(obj, list):
        return [convert_to_serializable(item) for item in obj]
    else:
        return obj  # Base case: return the object itself if it doesn't need conversion



def get_dm_displacement( command_vector, gain, sigma, X, Y, x0, y0 ):
    """_summary_

    Args:
        command_vector (_type_): _description_
        gain (float): command to opd for all actuators
        sigma (1D array): interactuator coupling, length= # actuators
        X (2D meshgrid): X coordinates of the space you want the DM to be in (e.g. pixel space)
        Y (2D meshgrid): Y coordinates of the space you want the DM to be in (e.g. pixel space)
        x0 (1D array): DM actuator x centers in X,Y space length= # actuators
        y0 (1D array): DM actuator y centers in X,Y space length= # actuators

    Returns:
        2D array: displancement map of DM in X,Y space
    """
    displacement_map = np.zeros( X.shape )
    for i in range( len( command_vector )):   
        #print(i)
        displacement_map += gaussian_displacement( c_i = gain * command_vector[i] , sigma_i=sigma[i], x=X, y=Y, x0=x0[i], y0=y0[i] )
    return displacement_map 



def get_pupil_intensity( phi, theta , phasemask, amp ): 
    """_summary_

    Args:
        phi (_type_): OPD (m)
        theta (_type_): phaseshift of mask (rad)
        phasemask ( ) : 2D array of the phaseshifting region in image plane 
            (Note: phi and amp implicitly have pupi geometry encoded in them for the PSF)
        amp (_type_): input amplitude of field

    Returns:
        _type_: ZWFS pupil intensity
    """

    psi_A = amp * np.exp( 1J * ( phi ) )

    psi_B = np.fft.fftshift(np.fft.fft2(np.fft.ifftshift( psi_A )) )
                            
    b = np.fft.fftshift( np.fft.ifft2( phasemask * psi_B ) )  

    psi_R = abs( b ) * np.sqrt((np.cos(theta)-1)**2 + np.sin(theta)**2)
    mu = np.angle((np.exp(1J*theta)-1) ) # np.arctan( np.sin(theta)/(np.cos(theta)-1) ) #

    # out formula ----------
    #if measured_pupil!=None:
    #    P = measured_pupil / np.mean( P[P > np.mean(P)] ) # normalize by average value in Pupil

    Ic = abs(psi_A)**2 + abs(psi_R)**2 + 2 * abs(psi_A) * abs(psi_R) * np.cos( phi - mu )  #+ beta)

    return Ic 



if __name__== "__main__":
    #############
    #### GRID 
    D = 1
    padding_factor = 4
    wvl0 = 1.65e-6 # m
    pupil, phasemask = get_grids( wavelength = wvl0 , F_number = 21.2, mask_diam = 1.06, diameter_in_angular_units = True, N = 2**6, padding_factor = padding_factor )
    theta = 3*np.pi/2 # rad  

    x = np.linspace( -(D*padding_factor)//2 ,(D*padding_factor)//2, pupil.shape[0] )
    y = np.linspace( -(D*padding_factor)//2 ,(D*padding_factor)//2, pupil.shape[0] )
    X, Y = np.meshgrid( x, y )


    #############
    #### DM 

    dm_pitch = 1   # DM actuator pitch in DM space
    a, b, c, d = D/np.ptp(x)/2, 0, 0, D/np.ptp(x)/2  # Parameters for affine transform (identity for simplicity)
    t_x, t_y = np.mean(x), np.mean(y)  # Translation in phase space

    transform_matrix = np.array( [[a,b,t_x],[c,d,t_y]] )

    dm_coords, dm_actuator_to_coord, dm_coord_to_actuator = generate_dm_coordinates(Nx=12, Ny=12, spacing=dm_pitch)

    #plt.figure(); plt.scatter([xx[0] for xx in dm_coords], [xx[1] for xx in dm_coords] ); plt.show(); 

    pixel_coord_list = np.array( [dm_to_pixel(c, transform_matrix) for c in dm_coords] )

    #plt.figure(); plt.scatter([xx[0] for xx in pixel_coord_list], [xx[1] for xx in pixel_coord_list] ); plt.show(); 

    command_vector = 0.5 + 0.001 * np.random.rand(140)
    sigma = 0.7 * (pixel_coord_list[0][0] - pixel_coord_list[1][0]) * np.ones( len( command_vector  ) ) # 1/2 actuator space in pixel space

    gain = 3e-6 # 3um per 1 dm cmd
    x0_list = [xx[0] for xx in pixel_coord_list]
    y0_list = [yy[1] for yy in pixel_coord_list]
    phi_dm = get_dm_displacement( command_vector=command_vector, gain=gain, sigma=sigma, X=X, Y=Y, x0=x0_list, y0=y0_list )

    plt.figure(); plt.imshow( phi_dm  ); plt.show() 


    #############
    #### FIELD 
    phi_atm = pupil * 10e-9 * np.random.randn( *pupil.shape)

    phi = 2*np.pi / wvl0 * ( phi_atm + phi_dm ) # phi_atm , phi_dm are in opd

    amp = 1e2 * pupil 

    N0 = get_pupil_intensity(  phi, theta = 0, phasemask=phasemask, amp=amp )

    I0 = get_pupil_intensity( phi= 0*phi, theta = np.pi/2, phasemask=phasemask, amp=amp )
    I0 *= np.sum( N0 ) / np.sum(I0)
    
    Ic =  get_pupil_intensity( phi, theta = np.pi/2, phasemask=phasemask, amp=amp )
    Ic *= np.sum( N0 ) / np.sum(Ic)
    
    
    fig = plt.figure() 
    im = plt.imshow( Ic, extent=[np.min(x), np.max(x), np.min(y), np.max(y)] )
    cbar = fig.colorbar(im, ax=plt.gca(), pad=0.01)
    cbar.set_label(r'$\Delta$ Intensity', fontsize=15, labelpad=10)

    plt.scatter(pixel_coord_list[:, 0], pixel_coord_list[:, 1], color='blue', marker='.', label = 'DM actuators')
    plt.show() 



    """
    psi_A = amp * np.exp( 1J * ( phi ) )

    psi_B = np.fft.fftshift(np.fft.fft2(np.fft.ifftshift( psi_A )) )
                            
    b = np.fft.fftshift( np.fft.ifft2( phasemask * psi_B ) ) 

    P = amp.copy() 

    psi_R = abs( b ) * np.sqrt((np.cos(theta)-1)**2 + np.sin(theta)**2)
    mu = np.angle((np.exp(1J*theta)-1) ) # np.arctan( np.sin(theta)/(np.cos(theta)-1) ) #

    # out formula ----------
    #if measured_pupil!=None:
    #    P = measured_pupil / np.mean( P[P > np.mean(P)] ) # normalize by average value in Pupil

    Ic =  abs(psi_A)**2 + abs(psi_R)**2 + 2 * abs(psi_A) * abs(psi_R) * np.cos( phi + mu )  #+ beta)
    """
    plt.figure(); plt.imshow( I0-N0 ); plt.show()
    plt.figure(); plt.imshow( Ic-N0 ); plt.show()
