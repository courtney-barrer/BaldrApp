import numpy as np
import matplotlib.pyplot as plt
from types import SimpleNamespace
import datetime 
from astropy.io import fits 
from . import utilities as util
from . import DM_basis as gen_basis


# PID and leaky integrator copied from /Users/bencb/Documents/asgard-alignment/playground/open_loop_tests_HO.py
class PIDController:
    def __init__(self, kp=None, ki=None, kd=None, upper_limit=None, lower_limit=None, setpoint=None):
        if kp is None:
            kp = np.zeros(1)
        if ki is None:
            ki = np.zeros(1)
        if kd is None:
            kd = np.zeros(1)
        if lower_limit is None:
            lower_limit = np.zeros(1)
        if upper_limit is None:
            upper_limit = np.ones(1)
        if setpoint is None:
            setpoint = np.zeros(1)

        self.kp = np.array(kp)
        self.ki = np.array(ki)
        self.kd = np.array(kd)
        self.lower_limit = np.array(lower_limit)
        self.upper_limit = np.array(upper_limit)
        self.setpoint = np.array(setpoint)
        self.ctrl_type = 'PID'
        
        size = len(self.kp)
        self.output = np.zeros(size)
        self.integrals = np.zeros(size)
        self.prev_errors = np.zeros(size)

    def process(self, measured):
        measured = np.array(measured)
        size = len(self.setpoint)

        if len(measured) != size:
            raise ValueError(f"Input vector size must match setpoint size: {size}")

        # Check all vectors have the same size
        error_message = []
        for attr_name in ['kp', 'ki', 'kd', 'lower_limit', 'upper_limit']:
            if len(getattr(self, attr_name)) != size:
                error_message.append(attr_name)
        
        if error_message:
            raise ValueError(f"Input vectors of incorrect size: {' '.join(error_message)}")

        if len(self.integrals) != size:
            print("Reinitializing integrals, prev_errors, and output to zero with correct size.")
            self.integrals = np.zeros(size)
            self.prev_errors = np.zeros(size)
            self.output = np.zeros(size)

        for i in range(size):
            error = measured[i] - self.setpoint[i]  # same as rtc
            self.integrals[i] += error
            self.integrals[i] = np.clip(self.integrals[i], self.lower_limit[i], self.upper_limit[i])

            derivative = error - self.prev_errors[i]
            self.output[i] = (self.kp[i] * error +
                              self.ki[i] * self.integrals[i] +
                              self.kd[i] * derivative)
            self.prev_errors[i] = error

        return self.output

    def set_all_gains_to_zero(self):
        self.kp = np.zeros( len(self.kp ))
        self.ki = np.zeros( len(self.ki ))
        self.kd = np.zeros( len(self.kd ))
        
    def reset(self):
        self.integrals.fill(0.0)
        self.prev_errors.fill(0.0)
        self.output.fill(0.0)
        
        

class LeakyIntegrator:
    def __init__(self, ki=None, lower_limit=None, upper_limit=None, kp=None):
        # If no arguments are passed, initialize with default values
        if ki is None:
            self.ki = []
            self.lower_limit = []
            self.upper_limit = []
            self.kp = []
        else:
            if len(ki) == 0:
                raise ValueError("ki vector cannot be empty.")
            if len(lower_limit) != len(ki) or len(upper_limit) != len(ki):
                raise ValueError("Lower and upper limit vectors must match ki vector size.")
            if kp is None or len(kp) != len(ki):
                raise ValueError("kp vector must be the same size as ki vector.")

            self.ki = np.array(ki)
            self.output = np.zeros(len(ki))
            self.lower_limit = np.array(lower_limit)
            self.upper_limit = np.array(upper_limit)
            self.kp = np.array(kp)  # kp is a vector now
        self.ctrl_type = 'Leaky'
    def process(self, input_vector):
        input_vector = np.array(input_vector)

        # Error checks
        if len(input_vector) != len(self.ki):
            raise ValueError("Input vector size must match ki vector size.")

        size = len(self.ki)
        error_message = ""

        if len(self.ki) != size:
            error_message += "ki "
        if len(self.lower_limit) != size:
            error_message += "lower_limit "
        if len(self.upper_limit) != size:
            error_message += "upper_limit "
        if len(self.kp) != size:
            error_message += "kp "

        if error_message:
            raise ValueError("Input vectors of incorrect size: " + error_message)

        if len(self.output) != size:
            print(f"output.size() != size.. reinitializing output to zero with correct size")
            self.output = np.zeros(size)

        # Process with the kp vector
        self.output = self.ki * self.output + self.kp * input_vector
        self.output = np.clip(self.output, self.lower_limit, self.upper_limit)

        return self.output


    def set_all_gains_to_zero(self):
        self.ki = np.zeros( len(self.ki ))
        self.kp = np.zeros( len(self.kp ))
        
        
    def reset(self):
        self.output = np.zeros(len(self.ki))

        

def reset_telemetry( zwfs_ns ):
    zwfs_ns.telem = SimpleNamespace(**init_telem_dict())
    return( zwfs_ns )
    

def reset_ctrl( zwfs_ns ):
    zwfs_ns.ctrl.TT_ctrl.reset()
    zwfs_ns.ctrl.HO_ctrl.reset()

    zwfs_ns.telem = init_telem_dict()




def init_telem_dict(): 
    
    telemetry_dict = {
        "i_list" : [],
        "s_list" : [],
        "e_TT_list" : [],
        "u_TT_list" : [],
        "c_TT_list" : [],
        "e_HO_list" : [],
        "u_HO_list" : [],
        "c_HO_list" : [],
        "atm_disturb_list" : [],
        "dm_disturb_list" : [],
        "rmse_list" : [],
        "flux_outside_pupil_list" : [],
        "residual_list" : [],
        "field_phase" : [],
        "strehl": []
    }
    return telemetry_dict


def save_telemetry( zwfs_ns , savename = None):
    
    tstamp = datetime.datetime.now().strftime("%d-%m-%YT%H.%M.%S")

    telem_dict = vars(zwfs_ns.telem )
    # Create a list of HDUs (Header Data Units)
    hdul = fits.HDUList()

    # Add each list to the HDU list as a new extension
    for list_name, data_list in telem_dict.items():
        # Convert list to numpy array for FITS compatibility
        data_array = np.array(data_list, dtype=float)  # Ensure it is a float array or any appropriate type

        # Create a new ImageHDU with the data
        hdu = fits.ImageHDU(data_array)

        # Set the EXTNAME header to the variable name
        hdu.header['EXTNAME'] = list_name

        # Append the HDU to the HDU list
        hdul.append(hdu)

    # Write the HDU list to a FITS file
    if savename is None:
        savename = f'~/Downloads/telemetry_simulation_{tstamp}.fits'
    hdul.writeto( savename, overwrite=True)





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
    #beta = np.angle( b )
    # out formula ----------
    #if measured_pupil!=None:
    #    P = measured_pupil / np.mean( P[P > np.mean(P)] ) # normalize by average value in Pupil

    Ic = abs(psi_A)**2 + abs(psi_R)**2 + 2 * abs(psi_A) * abs(psi_R) * np.cos( phi - mu )  #+ beta)

    return Ic 



"""def get_b_fresnel( phi, phasemask, wavelength, dx, z):
    k = 2 * np.pi / wavelength
    N = phi.shape[0]
    x = np.linspace(-N/2, N/2, N) * dx
    X, Y = np.meshgrid(x, x)

    # Initial field
    psi_A = np.exp(1j * phi)

    # Fresnel quadratic phase factor
    Q1 = np.exp(1j * k * (X**2 + Y**2) / (2 * z))

    # Apply Fresnel approximation for propagation to the focal plane
    psi_B = np.fft.fftshift(np.fft.fft2(np.fft.ifftshift(psi_A * Q1)))

    # Apply the phase mask
    b = np.fft.fftshift(np.fft.ifft2(phasemask * psi_B))

    return b


def get_b( phi, phasemask ):
    psi_A = np.exp( 1J * ( phi ) )

    psi_B = np.fft.fftshift(np.fft.fft2(np.fft.ifftshift( psi_A )) )
                            
    b = np.fft.fftshift( np.fft.ifft2( phasemask * psi_B ) )  
    
    return b"""

def update_dm_registration( transform_matrix, zwfs_ns ):
    """_summary_
    # STANDARD WAY TO UPDATE THE REGISTRATION OF THE DM IN WAVE SPACE 
    # UPDATES --> zwfs_ns <--- name space !!! Only use this method to update registration

    Args:
        transform_matrix (_type_): _description_ affine transform describing the mapping from DM actuators to the wavefront 
        zwfs_ns (_type_): _description_ the zwfs name space (holding configuration details)
        
    zwfs_ns dependancies (must have in namespace for code to work):
        zwfs_ns.dm.Nact_x  (int)
        zwfs_ns.dm.Nact_y  (int)
        zwfs_ns.dm.dm_pitch  (float)
        zwfs_ns.dm.actuator_coupling_factor (float)
        zwfs_ns.dm.current_cmd (1D array, size 140 for BMC multi-3.5 DM)
    
    """
    
    
    dm_coords, dm_actuator_to_coord, dm_coord_to_actuator = generate_dm_coordinates(Nx= zwfs_ns.dm.Nact_x , Ny= zwfs_ns.dm.Nact_y , spacing=zwfs_ns.dm.dm_pitch)

    #plt.figure(); plt.scatter([xx[0] for xx in dm_coords], [xx[1] for xx in dm_coords] ); plt.show(); 

    pixel_coord_list = np.array( [dm_to_pixel(c, transform_matrix) for c in dm_coords] )

    #plt.figure(); plt.scatter([xx[0] for xx in pixel_coord_list], [xx[1] for xx in pixel_coord_list] ); plt.show(); 

    # projecting the DM actuator space to wavespace. For convinience this is same as pixel space (before binnning)
    sigma = zwfs_ns.dm.actuator_coupling_factor * abs(pixel_coord_list[0][0] - pixel_coord_list[1][0]) * np.ones( len( zwfs_ns.dm.current_cmd  ) ) # coupling of actuators projected to wavespace

    x0_list = [xx[0] for xx in pixel_coord_list]
    y0_list = [yy[1] for yy in pixel_coord_list]
    

    dm2pixel_registration_dict = {
            "dm_to_pixsp_transform_matrix" : transform_matrix, # affine transform from DM coordinates to wave coordinates 
            "dm_actuator_to_coord" : dm_actuator_to_coord,
            "dm_coord_to_actuator" :dm_coord_to_actuator,
            }
    
    dm_coord_dict = {
        "dm_coords" : dm_coords, # DM coordinates DM space 
        "dm_coord_wavesp" : pixel_coord_list,
        "act_x0_list_wavesp" : x0_list, #actuator x coorindate in pixel space
        "act_y0_list_wavesp" : y0_list, #actuator y coordinate in pixel space
        "act_sigma_wavesp" : sigma
        
    }
    

    
    dm2pixel_registration_ns =  SimpleNamespace(**dm2pixel_registration_dict )
    #wave_coord_ns = SimpleNamespace(**wave_coord_dict )
    dm_coord_ns =  SimpleNamespace(**dm_coord_dict )
    
    # Add DM and wave coorindates to grid namespace 
    zwfs_ns.grid.dm_coord = dm_coord_ns
    zwfs_ns.dm2pixel_registration = dm2pixel_registration_ns
    
    return zwfs_ns
    
def init_ns_from_pyZelda(z, wvl0):
    grid_dict = {
    "D":8, # diameter of beam (m)
    "N" : z.pupil_diameter, # number of pixels across pupil diameter
    "padding_factor" : z.pupil_dim // z.pupil_diameter, # how many pupil diameters fit into grid x axis
    # TOTAL NUMBER OF PIXELS = padding_factor * N 
    }

    optics_dict = {
        "wvl0" :wvl0 , # central wavelength (m) 
        "F_number": z.mask_Fratio   , # F number on phasemask
        "mask_diam": z.mask_diameter / (1.22 * z.mask_Fratio * wvl0 ), # diameter of phaseshifting region in diffraction limit units (physical unit is mask_diam * 1.22 * F_number * lambda)
        "theta": z.mask_phase_shift( wvl0 )  # phaseshift of phasemask 
    }
    
    grid_ns = SimpleNamespace(**grid_dict)
    optics_ns = SimpleNamespace(**optics_dict)
    
    return grid_ns, optics_ns

def check_ns_consistency_with_pyZelda( z, zwfs_ns ):
    
    fail_log = {}
    
    if not z.pupil_diameter == zwfs_ns.grid.N:
        fail_log['pupil_diameter'] = (z.pupil_diameter, zwfs_ns.grid.N)
    if not z.pupil_dim // z.pupil_diameter == zwfs_ns.grid.padding_factor:
        fail_log['pupil_dim'] = (z.pupil_dim // z.pupil_diameter, zwfs_ns.grid.padding_factor)
    if not z.mask_diameter == zwfs_ns.optics.mask_diam * (1.22 * zwfs_ns.optics.F_number * zwfs_ns.optics.wvl0):
        fail_log["mask_diameter"] = (z.mask_diameter, zwfs_ns.optics.mask_diam * (1.22 * zwfs_ns.optics.F_number * zwfs_ns.optics.wvl0))   
    if not z.mask_Fratio == zwfs_ns.optics.F_number:
        fail_log["mask_Fratio"] = (z.mask_Fratio, zwfs_ns.optics.F_number)
    if not z.mask_phase_shift( zwfs_ns.optics.wvl0 ) == zwfs_ns.optics.theta:
        fail_log["theta"] = (z.mask_phase_shift( zwfs_ns.optics.wvl0 ), zwfs_ns.optics.theta)
    
    return fail_log
    

def init_zwfs(grid_ns, optics_ns, dm_ns):
    #############
    #### GRID 

    # get the pupil and phasemask masks
    pupil, phasemask = get_grids( wavelength = optics_ns.wvl0 , F_number = optics_ns.F_number, mask_diam = optics_ns.mask_diam, diameter_in_angular_units = True, N = grid_ns.N, padding_factor = grid_ns.padding_factor )

    grid_ns.pupil_mask = pupil
    grid_ns.phasemask_mask = phasemask
    
    # coorindates in the pupil plance
    x = np.linspace( -(grid_ns.D * grid_ns.padding_factor)//2 ,(grid_ns.D * grid_ns.padding_factor)//2, pupil.shape[0] )
    y = np.linspace( -(grid_ns.D * grid_ns.padding_factor)//2 ,(grid_ns.D * grid_ns.padding_factor)//2, pupil.shape[0] )
    X, Y = np.meshgrid( x, y )

    wave_coord_dict = {
            "x" : x,
            "y" : y, 
            "X" : X,
            "Y" : Y,
    }
    
    wave_coord_ns = SimpleNamespace(**wave_coord_dict )
    
    grid_ns.wave_coord = wave_coord_ns
    
    #############
    #### DM 

    if dm_ns.dm_model == "BMC-multi-3.5":
        # 12x12 with missing corners
        dm_ns.Nact_x = 12 #actuators along DM x axis
        dm_ns.Nact_y = 12 #actuators along DM y axis
        dm_ns.Nact = 140 # total number of actuators
        dm_ns.dm_flat = 0.5 + dm_ns.flat_rmse * np.random.rand(dm_ns.Nact) # add some noise to the DM flat 
        dm_ns.current_cmd = dm_ns.dm_flat # default set to dm flat 
    else:
        raise TypeError("input DM model not implemented. Try BMC-multi-3.5")
    
    # grid_ns.D/np.ptp(x)/padding_factor/2
    a, b, c, d = grid_ns.D/np.ptp(x)/2, 0, 0, grid_ns.D/np.ptp(x)/2  # Parameters for affine transform (identity for simplicity)
    # set by default to be centered and overlap with pupil (pupil touches edge of DM )
    
    t_x, t_y = np.mean(x), np.mean(y)  # Translation in phase space

    # we can introduce mis-registrations by rolling input pupil 
    
    dm_act_2_wave_space_transform_matrix = np.array( [[a,b,t_x],[c,d,t_y]] )

    
    # ZWFS NAME SPACE 
    zwfs_dict = {
        "grid":grid_ns,
        "optics":optics_ns,
        "dm":dm_ns,
        #"dm2pixel_registration" : dm2pixel_registration_ns
        }
        
    zwfs_ns = SimpleNamespace(**zwfs_dict)
    
    # user warning: only ever use update_dm_registration if you want a consistent update across all variables 
    # this updates the zwfs_ns.grid with dm coords in DM and wavespace as well as defining dm2pixel_registration namespace
    zwfs_ns = update_dm_registration(  dm_act_2_wave_space_transform_matrix, zwfs_ns )
    
    return zwfs_ns 



def test_propagation( zwfs_ns ):
    """_summary_
    just test propagating through the zwfs system with :
        -small (10nm rms) random internal aberations , 
        -no atmospheric aberations ,
        -the current DM state in the zwfs_ns
    Args:
        zwfs_ns (_type_): _description_

    Returns:
        _type_: phi, phi_internal, N0, I0, I
            phi is the wavefront phase (at defined central wvl) from current dm, atm, internal aberrations 
            phi_internal is the wavefront phase (at defined central wvl)  from defined flat dm, internal aberrations 
            N0 is intensity with flat dm, no phasemask
            I0 is intensity with flat dm,  phasemask
            I is intensity with input dm, phasemask
    """
    opd_atm = 0 * zwfs_ns.grid.pupil_mask *  np.random.randn( *zwfs_ns.grid.pupil_mask.shape)

    opd_internal = 10e-9 * zwfs_ns.grid.pupil_mask * np.random.randn( *zwfs_ns.grid.pupil_mask.shape)

    # get the OPD from the DM in the wave space.
    # the only real dynamic thing needed is the current command of the DM 
    # zwfs_ns.dm.current_cmd
    opd_flat_dm = get_dm_displacement( command_vector= zwfs_ns.dm.dm_flat , gain=zwfs_ns.dm.opd_per_cmd, \
        sigma= zwfs_ns.grid.dm_coord.act_sigma_wavesp, X=zwfs_ns.grid.wave_coord.X, Y=zwfs_ns.grid.wave_coord.Y,\
            x0=zwfs_ns.grid.dm_coord.act_x0_list_wavesp, y0=zwfs_ns.grid.dm_coord.act_y0_list_wavesp )

    opd_current_dm = get_dm_displacement( command_vector= zwfs_ns.dm.current_cmd   , gain=zwfs_ns.dm.opd_per_cmd, \
        sigma= zwfs_ns.grid.dm_coord.act_sigma_wavesp, X=zwfs_ns.grid.wave_coord.X, Y=zwfs_ns.grid.wave_coord.Y,\
            x0=zwfs_ns.grid.dm_coord.act_x0_list_wavesp, y0=zwfs_ns.grid.dm_coord.act_y0_list_wavesp )


    phi_internal = 2*np.pi / zwfs_ns.optics.wvl0 * ( opd_internal + opd_flat_dm  ) # phi_atm , phi_dm are in opd

    phi = 2*np.pi / zwfs_ns.optics.wvl0 * ( opd_internal + opd_current_dm + opd_atm ) # phi_atm , phi_dm are in opd

    amp = 1e2 *zwfs_ns.grid.pupil_mask

    N0 = get_pupil_intensity(  phi = phi_internal, theta = 0, phasemask=zwfs_ns.grid.phasemask_mask, amp=amp )

    I0 = get_pupil_intensity( phi = phi_internal, theta =zwfs_ns.optics.theta, phasemask=zwfs_ns.grid.phasemask_mask, amp=amp )

    Intensity = get_pupil_intensity( phi = phi, theta = zwfs_ns.optics.theta, phasemask=zwfs_ns.grid.phasemask_mask, amp=amp )

    return phi, phi_internal, N0, I0, Intensity 


def average_subarrays(array, block_size):
    """
    Averages non-overlapping sub-arrays of a given 2D NumPy array.
    
    Parameters:
    array (numpy.ndarray): Input 2D array of shape (N, M).
    block_size (tuple): Size of the sub-array blocks (height, width).
    
    Returns:
    numpy.ndarray: 2D array containing the averaged values of the sub-arrays.
    """
    # Check if the array dimensions are divisible by the block size
    if array.shape[0] % block_size[0] != 0 or array.shape[1] % block_size[1] != 0:
        raise ValueError("Array dimensions must be divisible by the block size.")
    
    # Reshape the array to isolate the sub-arrays
    reshaped = array.reshape(array.shape[0] // block_size[0], block_size[0], 
                             array.shape[1] // block_size[1], block_size[1])
    
    # Compute the mean of the sub-arrays
    averaged_subarrays = reshaped.mean(axis=(1, 3))
    
    return averaged_subarrays

    
    
def get_I0(  opd_input,  amp_input , opd_internal,  zwfs_ns , detector=None):
    # get intensity with the phase mask in the beam
    # forces dm to be in the current flat configuration (zwfs_ns.dm.dm_flat )
    # FOR NOW detector = None or a tuple representing the binning to perform on intensity in wavespace. add noise etc . Later consider claass for this
    opd_current_dm = get_dm_displacement( command_vector = zwfs_ns.dm.dm_flat  , gain=zwfs_ns.dm.opd_per_cmd, \
        sigma= zwfs_ns.grid.dm_coord.act_sigma_wavesp, X=zwfs_ns.grid.wave_coord.X, Y=zwfs_ns.grid.wave_coord.Y,\
            x0=zwfs_ns.grid.dm_coord.act_x0_list_wavesp, y0=zwfs_ns.grid.dm_coord.act_y0_list_wavesp )
    
    phi = zwfs_ns.grid.pupil_mask  *  2*np.pi / zwfs_ns.optics.wvl0 * (  opd_input + opd_internal + opd_current_dm  )

    Intensity = get_pupil_intensity( phi = phi, theta = zwfs_ns.optics.theta, phasemask=zwfs_ns.grid.phasemask_mask, amp=amp_input )

    if detector is not None:
        Intensity = average_subarrays(array=Intensity, block_size = detector)
        

    return Intensity

def get_N0( opd_input,  amp_input ,  opd_internal,  zwfs_ns , detector=None):
    # get intensity with the phase mask out of the beam (here we jiust put theta = 0)
    # forces dm to be in the current flat configuration (zwfs_ns.dm.dm_flat )
    # FOR NOW detector = None or a tuple representing the binning to perform on intensity in wavespace. add noise etc 
    opd_current_dm = get_dm_displacement( command_vector= zwfs_ns.dm.dm_flat   , gain=zwfs_ns.dm.opd_per_cmd, \
        sigma= zwfs_ns.grid.dm_coord.act_sigma_wavesp, X=zwfs_ns.grid.wave_coord.X, Y=zwfs_ns.grid.wave_coord.Y,\
            x0=zwfs_ns.grid.dm_coord.act_x0_list_wavesp, y0=zwfs_ns.grid.dm_coord.act_y0_list_wavesp )
    
    phi = zwfs_ns.grid.pupil_mask  *  2*np.pi / zwfs_ns.optics.wvl0 * ( opd_input + opd_internal + opd_current_dm  )

    
    Intensity = get_pupil_intensity( phi = phi, theta = 0, phasemask=zwfs_ns.grid.phasemask_mask, amp=amp_input )

    if detector is not None:
        Intensity = average_subarrays(array=Intensity, block_size = detector)

    return Intensity


def get_frame( opd_input,  amp_input ,  opd_internal,  zwfs_ns , detector=None):
    # get intensity with the phase mask in the beam
    # and dm shaped to the current command ( zwfs_ns.dm.current_cmd ) 
    # FOR NOW detector = None or a tuple representing the binning to perform on intensity in wavespace. add noise etc 
    opd_current_dm = get_dm_displacement( command_vector= zwfs_ns.dm.current_cmd   , gain=zwfs_ns.dm.opd_per_cmd, \
        sigma= zwfs_ns.grid.dm_coord.act_sigma_wavesp, X=zwfs_ns.grid.wave_coord.X, Y=zwfs_ns.grid.wave_coord.Y,\
            x0=zwfs_ns.grid.dm_coord.act_x0_list_wavesp, y0=zwfs_ns.grid.dm_coord.act_y0_list_wavesp )
    
    phi = zwfs_ns.grid.pupil_mask  *  2*np.pi / zwfs_ns.optics.wvl0 * ( opd_input + opd_internal + opd_current_dm  )
    
    Intensity = get_pupil_intensity( phi = phi, theta = zwfs_ns.optics.theta, phasemask=zwfs_ns.grid.phasemask_mask, amp=amp_input )

    if detector is not None:
        Intensity = average_subarrays(array=Intensity, block_size = detector)

    return Intensity


def classify_pupil_regions( opd_input,  amp_input ,  opd_internal,  zwfs_ns , detector=None, pupil_diameter_scaling = 1.0, pupil_offset = (0,0)):
    # very basic pupil classification
    # adds to zwfs_ns 
    # inside pupil 
    
    # We intentionally put detector as None here to keep intensities in wave space
    # we do the math here and then bin after if user selects detector is not None    
    N0 = get_N0( opd_input,  amp_input ,  opd_internal,  zwfs_ns , detector=None)
    I0 = get_I0( opd_input,  amp_input ,  opd_internal,  zwfs_ns , detector=None)
    
    # currently we don't use N0 to classify, just use known pupil diameter 
    #pupil_filt = zwfs_ns.grid.pupil_mask > 0.5
    pupil_filt = (zwfs_ns.grid.wave_coord.X - pupil_offset[0])**2 + (zwfs_ns.grid.wave_coord.Y - pupil_offset[1])**2 <= pupil_diameter_scaling * (zwfs_ns.grid.D/2)**2

    outside_filt = ~pupil_filt 
    
    secondary_strehl_filt = (zwfs_ns.grid.wave_coord.X - pupil_offset[0])**2 + (zwfs_ns.grid.wave_coord.Y - pupil_offset[1])**2 < (zwfs_ns.grid.D/10)**2
    
    outer_strehl_filt = ( I0 - N0 >   4.5 * np.median(I0) ) * outside_filt
    
    if detector is not None:
        pupil_filt = average_subarrays(array= pupil_filt, block_size=detector) > 0
        
        outside_filt = average_subarrays(array= outside_filt, block_size=detector) > 0
        
        secondary_strehl_filt = average_subarrays( secondary_strehl_filt ,block_size=detector) > 0
        
        outer_strehl_filt = average_subarrays( outer_strehl_filt ,block_size=detector) > 0


    region_classification_dict = {
        "pupil_filt":pupil_filt,
        "outside_filt":outside_filt,
        "secondary_strehl_filt":secondary_strehl_filt,
        "outer_strehl_filt":outer_strehl_filt }
    
    regions_ns = SimpleNamespace(**region_classification_dict ) 
    
    zwfs_ns.pupil_regions = regions_ns
    
    return( zwfs_ns )


def process_zwfs_signal( I, I0, pupil_filt ): 
    """_summary_
    
    STANDARD WAY TO PROCESS ZWFS ERROR SIGNAL FROM INTENSITY MEASUREMENT 

    Args:
        I (_type_): _description_
        I0 (_type_): _description_
        pupil_filt (_type_): _description_

    Returns:
        _type_: _description_
    """
    s = ( I / np.mean( I ) -  I0 / np.mean( I0 ) )[pupil_filt] 
    return s.reshape(-1) 


    
def build_IM( zwfs_ns ,  calibration_opd_input, calibration_amp_input ,  opd_internal,  basis = 'Zonal_pinned', Nmodes = 100, poke_amp = 0.05, poke_method = 'double_sided_poke', imgs_to_mean = 10, detector=None):
    
    # build reconstructor name space with normalized basis, IM generated, IM generation method, pokeamp 
    modal_basis = gen_basis.construct_command_basis( basis= basis, number_of_modes = Nmodes, without_piston=True).T 

    IM=[] # init our raw interaction matrix 

    I0_list = []
    for _ in range(imgs_to_mean) :
        I0_list .append( get_I0( opd_input  =calibration_opd_input,    amp_input = calibration_amp_input  ,  opd_internal= opd_internal,  zwfs_ns=zwfs_ns , detector=detector )  )
    I0 = np.mean( I0_list ,axis =0 )
    
    N0_list = []
    for _ in range(imgs_to_mean) :
        N0_list .append( get_N0( opd_input  =calibration_opd_input,    amp_input = calibration_amp_input  ,  opd_internal= opd_internal,  zwfs_ns=zwfs_ns , detector=detector )  )
    N0 = np.mean( N0_list ,axis =0 )
    
    if poke_method=='single_sided_poke': # just poke one side  
                

        for i,m in enumerate(modal_basis):
            print(f'executing cmd {i}/{len(modal_basis)}')       
                
            zwfs_ns.dm.current_cmd = zwfs_ns.dm.dm_flat + poke_amp * m
            
            img_list = []
            for _ in range( imgs_to_mean ):
                img_list.append( get_frame( calibration_opd_input,   calibration_amp_input  ,  opd_internal,  zwfs_ns , detector=detector ) ) # get some frames 
                 
            Intensity = np.mean( img_list, axis = 0).reshape(-1) 

            # IMPORTANT : we normalize by mean over total image region (post reduction) (NOT FILTERED )... 
            Intensity *= 1/np.mean( Intensity ) # we normalize by mean over total region! 
            
            # get intensity error signal 
            errsig = process_zwfs_signal( Intensity, I0, zwfs_ns.pupil_regions.pupil_filt )

            IM.append( list(  errsig.reshape(-1) ) ) #toook out 1/poke_amp *

    elif poke_method=='double_sided_poke':
        for i,m in enumerate(modal_basis):
            print(f'executing cmd {i}/{len(modal_basis)}')
            I_plus_list = []
            I_minus_list = []
            for sign in [(-1)**n for n in range(np.max([2, imgs_to_mean]))]: #[-1,1]:
                
                #ZWFS.dm.send_data( list( ZWFS.dm_shapes['flat_dm'] + sign * poke_amp/2 * m )  )
                zwfs_ns.dm.current_cmd = zwfs_ns.dm.dm_flat + sign * poke_amp/2 * m
                
                if sign > 0:
                    
                    I_plus_list += [list( get_frame( calibration_opd_input,   calibration_amp_input  ,  opd_internal,  zwfs_ns , detector=detector ) ) ]
                    
                if sign < 0:
                    
                    I_minus_list += [list( get_frame( calibration_opd_input,   calibration_amp_input  ,  opd_internal,  zwfs_ns , detector=detector ) ) ] 
                    

            I_plus = np.mean( I_plus_list, axis = 0).reshape(-1)  # flatten so can filter with ZWFS.pupil_pixels
            I_plus *= 1/np.mean( I_plus )

            I_minus = np.mean( I_minus_list, axis = 0).reshape(-1)  # flatten so can filter with ZWFS.pupil_pixels
            I_minus *= 1/np.mean( I_minus )

            errsig =  (I_plus - I_minus)[np.array( zwfs_ns.pupil_regions.pupil_filt.reshape(-1) )]
            IM.append( list(  errsig.reshape(-1) ) ) #toook out 1/poke_amp *

    else:
        raise TypeError( ' no matching method for building control model. Try (for example) method="single_side_poke"')

    # convert to array 
    IM = np.array( IM )  
    
    reco_dict = {
        "I0":I0,
        "N0":N0,
        "M2C_0":modal_basis,
        "basis_name":basis,
        "poke_amp":poke_amp,
        "poke_method":poke_method,
        "IM":IM,
    }
    
    reco_ns = SimpleNamespace(**reco_dict)
    
    zwfs_ns.reco = reco_ns
    
    return zwfs_ns



def plot_eigenmodes( zwfs_ns , save_path = None, descr_label=None ):
    
    tstamp = datetime.datetime.now().strftime("%d-%m-%YT%H.%M.%S")

    U,S,Vt = np.linalg.svd( zwfs_ns.reco.IM, full_matrices=True)

    #singular values
    plt.figure(1) 
    plt.semilogy(S) #/np.max(S))
    #plt.axvline( np.pi * (10/2)**2, color='k', ls=':', label='number of actuators in pupil')
    plt.legend() 
    plt.xlabel('mode index')
    plt.ylabel('singular values')

    if save_path is not None:
        plt.savefig(save_path +  f'singularvalues_{descr_label}_{tstamp}.png', bbox_inches='tight', dpi=200)
    #plt.show()
    
    # THE IMAGE MODES 
    n_row = round( np.sqrt( zwfs_ns.reco.M2C_0.shape[0]) ) - 1
    fig ,ax = plt.subplots(n_row  ,n_row ,figsize=(30,30))
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
        plt.savefig(save_path + f'det_eignmodes_{descr_label}_{tstamp}.png',bbox_inches='tight',dpi=200)
    #plt.show()
    
    # THE DM MODES 

    # NOTE: if not zonal (modal) i might need M2C to get this to dm space 
    # if zonal M2C is just identity matrix. 
    fig,ax = plt.subplots(n_row, n_row, figsize=(30,30))
    plt.subplots_adjust(hspace=0.1,wspace=0.1)
    for i,axx in enumerate(ax.reshape(-1)):
        axx.imshow( util.get_DM_command_in_2D( zwfs_ns.reco.M2C_0.T @ U.T[i] ) )
        #axx.set_title(f'mode {i}, S={round(S[i]/np.max(S),3)}')
        axx.text( 1,2,f'{i}',color='w',fontsize=6)
        axx.text( 1,3,f'S={round(S[i]/np.max(S),3)}',color='w',fontsize=6)
        axx.axis('off')
        #plt.legend(ax=axx)
    plt.tight_layout()
    if save_path is not None:
        plt.savefig(save_path +  f'dm_eignmodes_{descr_label}_{tstamp}.png',bbox_inches='tight',dpi=200)
    #plt.show()


def construct_ctrl_matricies_from_IM(zwfs_ns,  method = 'Eigen_TT-HO', Smax = 50, TT_vectors = gen_basis.get_tip_tilt_vectors() ):


    M2C_0 = zwfs_ns.reco.M2C_0
    poke_amp = zwfs_ns.reco.poke_amp 
    
    if method == 'Eigen_TT-HO':    
        U, S, Vt = np.linalg.svd( zwfs_ns.reco.IM, full_matrices=False)

        R  = (Vt.T * [1/ss if i < Smax else 0 for i,ss in enumerate(S)])  @ U.T

        #TT_vectors = gen_basis.get_tip_tilt_vectors()

        TT_space = M2C_0 @ TT_vectors
            
        U_TT, S_TT, Vt_TT = np.linalg.svd( TT_space, full_matrices=False)

        I2M_TT = U_TT.T @ R.T 

        M2C_TT = poke_amp * M2C_0.T @ U_TT # since pinned need M2C to go back to 140 dimension vector  

        R_HO = (np.eye(U_TT.shape[0])  - U_TT @ U_TT.T) @ R.T

        # go to Eigenmodes for modal control in higher order reconstructor
        U_HO, S_HO, Vt_HO = np.linalg.svd( R_HO, full_matrices=False)
        I2M_HO = Vt_HO  
        M2C_HO = poke_amp *  M2C_0.T @ (U_HO * S_HO) # since pinned need M2C to go back to 140 dimension vector
        # plt.plot( M2C_HO @ I2M_HO @ IM[63] ) ; plt.show()
        reco_dict_current = vars( zwfs_ns.reco )
        reco_dict_2append = {
            "U":U,
            "S":S,
            "Vt":Vt,
            "Smax":Smax,
            "R":R,
            "U_TT":U_TT,
            "S_TT":S_TT,
            "Vt_TT":Vt_TT,
            "U_HO":U_HO,
            "S_HO":S_HO,
            "Vt_HO":Vt_HO,
            "I2M_TT":I2M_TT,
            "M2C_TT":M2C_TT,
            "I2M_HO":I2M_HO,
            "M2C_HO":M2C_HO
            }
        reco_dict = reco_dict_current | reco_dict_2append
        zwfs_ns.reco = SimpleNamespace( **reco_dict  )# add it to the current reco namespace with 
        
    elif method == 'Eigen_HO': # just look at eigen basis in HO - no projection onto TT
        U, S, Vt = np.linalg.svd( zwfs_ns.reco.IM.T, full_matrices=False)

        R  = (Vt.T * [1/ss if i < Smax else 0 for i,ss in enumerate(S)])  @ U.T

        # 
        TT_space = M2C_0 @ TT_vectors
        U_TT, S_TT, Vt_TT = np.linalg.svd( TT_space, full_matrices=False)

        I2M_TT = np.zeros( [2, R.shape[1]] ) # set to zero - we only consider HO in eigenmodes 
        M2C_TT = TT_vectors

        U_HO , S_HO , Vt_HO = U.copy(), S.copy() , Vt.copy() 
        I2M_HO =  U.T # R.T
        M2C_HO = poke_amp *  M2C_0.T @ (Vt.T * [1/ss if i < Smax else 0 for i,ss in enumerate(S)]) 
        #plt.plot( M2C_HO @ I2M_HO @ IM[63] ) ; plt.show()
    else:
        raise TypeError('construct_ctrl_matricies_from_IM method name NOT FOUND!!!!')
        
    reco_dict_current = vars( zwfs_ns.reco )
    reco_dict_2append = {
        "U":U,
        "S":S,
        "Vt":Vt,
        "Smax":Smax,
        "R":R,
        "U_TT":U_TT,
        "S_TT":S_TT,
        "Vt_TT":Vt_TT,
        "U_HO":U_HO,
        "S_HO":S_HO,
        "Vt_HO":Vt_HO,
        "I2M_TT":I2M_TT,
        "M2C_TT":M2C_TT,
        "I2M_HO":I2M_HO,
        "M2C_HO":M2C_HO
        }
    reco_dict = reco_dict_current | reco_dict_2append
    zwfs_ns.reco = SimpleNamespace( **reco_dict  )# add it to the current reco namespace with 
    
    return zwfs_ns


def add_controllers( zwfs_ns , TT='PID', HO = 'leaky'):
 
    if HO == 'leaky':
        ki_leak = 0 * np.ones( zwfs_ns.reco.I2M_HO.shape[0] )
        kp_leak = 0 * np.ones( zwfs_ns.reco.I2M_HO.shape[0] )
        lower_limit_leak = -100 * np.ones( zwfs_ns.reco.I2M_HO.shape[0] )
        upper_limit_leak = 100 * np.ones( zwfs_ns.reco.I2M_HO.shape[0] )

        HO_ctrl = LeakyIntegrator(ki=ki_leak, kp=kp_leak, lower_limit=lower_limit_leak, upper_limit=upper_limit_leak )
    
    elif HO == 'PID':    
        kp = 0 * np.ones( zwfs_ns.reco.I2M_HO.shape[0] )
        ki = 0 * np.ones( zwfs_ns.reco.I2M_HO.shape[0] )
        kd = 0. * np.ones( zwfs_ns.reco.I2M_HO.shape[0] )
        setpoint = np.zeros( zwfs_ns.reco.I2M_HO.shape[0] )
        lower_limit_pid = -100 * np.ones( zwfs_ns.reco.I2M_HO.shape[0] )
        upper_limit_pid = 100 * np.ones( zwfs_ns.reco.I2M_HO.shape[0] )

        HO_ctrl = PIDController(kp, ki, kd, upper_limit_pid, lower_limit_pid, setpoint)

    if TT == 'leaky':
        ki = 0 * np.ones( zwfs_ns.reco.I2M_TT.shape[0] )
        kp_leak = 0 * np.ones( zwfs_ns.reco.I2M_TT.shape[0] )
        lower_limit_leak = -100 * np.ones( zwfs_ns.reco.I2M_TT.shape[0] )
        upper_limit_leak = 100 * np.ones( zwfs_ns.reco.I2M_TT.shape[0] )

        TT_ctrl = LeakyIntegrator(ki=ki, kp=kp_leak, lower_limit=lower_limit_leak, upper_limit=upper_limit_leak )
        
    elif TT == 'PID':
        kp = 0 * np.ones( zwfs_ns.reco.I2M_TT.shape[0] )
        ki = 0 * np.ones( zwfs_ns.reco.I2M_TT.shape[0] )
        kd = 0. * np.ones( zwfs_ns.reco.I2M_TT.shape[0] )
        setpoint = np.zeros( zwfs_ns.reco.I2M_TT.shape[0] )
        lower_limit_pid = -100 * np.ones( zwfs_ns.reco.I2M_TT.shape[0] )
        upper_limit_pid = 100 * np.ones( zwfs_ns.reco.I2M_TT.shape[0] )

        TT_ctrl = PIDController(kp, ki, kd, upper_limit_pid, lower_limit_pid, setpoint)

    controller_dict = {
        "TT_ctrl" : TT_ctrl,
        "HO_ctrl" : HO_ctrl
    }

    telemetry_dict = init_telem_dict()

    control_ns = SimpleNamespace(**controller_dict)
    tele_ns = SimpleNamespace(**telemetry_dict)

    zwfs_ns.ctrl = control_ns
    zwfs_ns.telem = tele_ns

    zwfs_ns.dm.current_cmd = zwfs_ns.dm.dm_flat

    return zwfs_ns


def AO_iteration( opd_input, amp_input, opd_internal, I0,  zwfs_ns, dm_disturbance = np.zeros(140), record_telemetry=True, detector=None, obs_intermediate_field=True):
    # single iteration of AO in closed loop 
    
    # propagates opd over DM and get intensity

    if obs_intermediate_field: # we onbserve the actual field (only valid in simulation to test results)
        # opd in wavespace 
        opd_current_dm = get_dm_displacement( command_vector= zwfs_ns.dm.current_cmd   , gain=zwfs_ns.dm.opd_per_cmd, \
            sigma= zwfs_ns.grid.dm_coord.act_sigma_wavesp, X=zwfs_ns.grid.wave_coord.X, Y=zwfs_ns.grid.wave_coord.Y,\
                x0=zwfs_ns.grid.dm_coord.act_x0_list_wavesp, y0=zwfs_ns.grid.dm_coord.act_y0_list_wavesp )
        
        phi = zwfs_ns.grid.pupil_mask  *  2*np.pi / zwfs_ns.optics.wvl0 * ( opd_input + opd_internal + opd_current_dm  )
        
        i = get_pupil_intensity( phi = phi, theta = zwfs_ns.optics.theta, phasemask=zwfs_ns.grid.phasemask_mask, amp=amp_input )

        if detector is not None:
            i = average_subarrays(array=i, block_size = detector)
        strehl = np.exp( -np.var( phi[ zwfs_ns.grid.pupil_mask > 0]) ) 
        
    else: # we just see intensity 
        i = get_frame(  opd_input  = opd_input ,   amp_input = amp_input,\
        opd_internal = opd_internal,  zwfs_ns= zwfs_ns , detector= detector )

    sig = process_zwfs_signal( i, I0, zwfs_ns.pupil_regions.pupil_filt ) # I0_theory/ np.mean(I0_theory) #

    e_TT = zwfs_ns.reco.I2M_TT @ sig

    u_TT = zwfs_ns.ctrl.TT_ctrl.process( e_TT )

    c_TT = zwfs_ns.reco.M2C_TT @ u_TT 

    e_HO = zwfs_ns.reco.I2M_HO @ sig

    u_HO = zwfs_ns.ctrl.HO_ctrl.process( e_HO )

    c_HO = zwfs_ns.reco.M2C_HO @ u_HO 

    # safety 
    if np.max( c_TT + c_HO ) > 0.8: 
        print( ' going badly.. ')
        
    # SEND DM COMMAND 
    zwfs_ns.dm.current_cmd =  zwfs_ns.dm.dm_flat +  dm_disturbance - c_HO - c_TT

    # only measure residual in the registered pupil on DM 
    residual =  (dm_disturbance - c_HO - c_TT)
    rmse = np.nanstd( residual )
     

    # telemetry 
    if record_telemetry :
        zwfs_ns.telem.i_list.append( i )
        zwfs_ns.telem.s_list.append( sig )
        zwfs_ns.telem.e_TT_list.append( e_TT )
        zwfs_ns.telem.u_TT_list.append( u_TT )
        zwfs_ns.telem.c_TT_list.append( c_TT )

        zwfs_ns.telem.e_HO_list.append( e_HO )
        zwfs_ns.telem.u_HO_list.append( u_HO )
        zwfs_ns.telem.c_HO_list.append( c_HO )

        #atm_disturb_list.append( scrn.scrn )
        zwfs_ns.telem.dm_disturb_list.append( dm_disturbance )

        zwfs_ns.telem.residual_list.append( residual )
        zwfs_ns.telem.rmse_list.append( rmse )
        
        if obs_intermediate_field:
            zwfs_ns.telem.field_phase.append( phi )
            zwfs_ns.telem.strehl.append( strehl )
            




# #### 

# grid_dict = {
#     "D":1, # diameter of beam 
#     "N" : 64, # number of pixels across pupil
#     "padding_factor" : 4, # how many pupil diameters fit into grid x axis
#     }

# optics_dict = {
#     "wvl0" :1.65e-6, # central wavelength (m) 
#     "F_number": 21.2, # F number on phasemask
#     "mask_diam": 1.06, # diameter of phaseshifting region in diffraction limit units (physical unit is mask_diam * 1.22 * F_number * lambda)
#     "theta": 1.57079, # phaseshift of phasemask 
# }

# dm_dict = {
#     "dm_model":"BMC-multi-3.5",
#     "actuator_coupling_factor":0.7,# std of in actuator spacing of gaussian IF applied to each actuator. (e.g actuator_coupling_factor = 1 implies std of poke is 1 actuator across.)
#     "dm_pitch":1,
#     "dm_aoi":0, # angle of incidence of light on DM 
#     "opd_per_cmd" : 3e-6, # peak opd applied at center of actuator per command unit (normalized between 0-1) 
#     "flat_rmse" : 20e-9 # std (m) of flatness across Flat DM  
#     }

# grid_ns = SimpleNamespace(**grid_dict)
# optics_ns = SimpleNamespace(**optics_dict)
# dm_ns = SimpleNamespace(**dm_dict)

# ################## TEST 1
# # check dm registration on pupil (wavespace)
# zwfs_ns = init_zwfs(grid_ns, optics_ns, dm_ns)

# opd_atm, opd_internal, opd_dm, phi,  N0, I0, I = test_propagation( zwfs_ns )

# fig = plt.figure() 
# im = plt.imshow( N0, extent=[np.min(zwfs_ns.grid.wave_coord.x), np.max(zwfs_ns.grid.wave_coord.x),\
#     np.min(zwfs_ns.grid.wave_coord.y), np.max(zwfs_ns.grid.wave_coord.y)] )
# cbar = fig.colorbar(im, ax=plt.gca(), pad=0.01)
# cbar.set_label(r'Pupil Intensity', fontsize=15, labelpad=10)

# plt.scatter(zwfs_ns.grid.dm_coord.act_x0_list_wavesp, zwfs_ns.grid.dm_coord.act_y0_list_wavesp, color='blue', marker='.', label = 'DM actuators')
# plt.show() 

# ################## TEST 2 
# # test updating the DM registration 
# zwfs_ns = init_zwfs(grid_ns, optics_ns, dm_ns)

# a, b, c, d = zwfs_ns.grid.D/np.ptp(zwfs_ns.grid.wave_coord.x)/2, 0, 0, grid_ns.D/np.ptp(zwfs_ns.grid.wave_coord.x)/2  # Parameters for affine transform (identity for simplicity)
# # set by default to be centered and overlap with pupil (pupil touches edge of DM )

# # offset 5% of pupil 
# t_x, t_y = np.mean(zwfs_ns.grid.wave_coord.x) + 0.05 * zwfs_ns.grid.D, np.mean(zwfs_ns.grid.wave_coord.x)  # Translation in phase space

# # we could also introduce mis-registrations by rolling input pupil 
# dm_act_2_wave_space_transform_matrix = np.array( [[a,b,t_x],[c,d,t_y]] )

# zwfs_ns = update_dm_registration( dm_act_2_wave_space_transform_matrix , zwfs_ns )

# opd_atm, opd_internal, opd_dm,  N0, I0, I = test_propagation( zwfs_ns )

# # dm in dm coords
# fig,ax = plt.subplots( 1,4 )
# ax[0].imshow( util.get_DM_command_in_2D( zwfs_ns.dm.current_cmd ))
# ax[0].set_title('dm cmd')
# ax[1].set_title('OPD wavespace')
# ax[1].imshow( phi )
# ax[2].set_title('ZWFS Intensity')
# ax[2].imshow( I )
# ax[3].set_title('ZWFS reference Intensity')
# ax[3].imshow( I0 )





# # phi in wave coords
 

#     field = propagate_field(input_field, zwfs_ns )
    
#     I = get_frame( field ) 
    
#     S = process_signal( I, zwfs_ns ) # add N0, I0, dm shape to zwfs_ns
    
#     #for each reconstructor space i
#     e_i = S @ R_i 
    
#     u_i = controller( e_i )
    
#     c_i = M2C @ u_i 
    
#     c = sum_i( c_i )
    
#     send_cmd( c ) <- updates zwfs_ns.dm_ns.dm_shape
    
    
        
    
    
    
#     phi_dm = get_dm_displacement( command_vector=command_vector, gain=dm_ns.opd_per_cmd, sigma= sigma, X=X, Y=Y, x0=x0_list, y0=y0_list )

#     #plt.figure(); plt.imshow( phi_dm  ); plt.show() 


#     #############
#     #### FIELD 
#     phi_atm = pupil * 10e-9 * np.random.randn( *pupil.shape)

#     phi = 2*np.pi / optics_ns.wvl0 * ( phi_atm + phi_dm ) # phi_atm , phi_dm are in opd

#     amp = 1e2 * pupil 

#     N0 = get_pupil_intensity(  phi, theta = 0, phasemask=phasemask, amp=amp )

#     I0 = get_pupil_intensity( phi= 0*phi, theta = np.pi/2, phasemask=phasemask, amp=amp )
#     I0 *= np.sum( N0 ) / np.sum(I0)
    
#     Ic =  get_pupil_intensity( phi, theta = np.pi/2, phasemask=phasemask, amp=amp )
#     Ic *= np.sum( N0 ) / np.sum(Ic)
    
#     ## SOME PLOTS 
#     #fig = plt.figure() 
#     #im = plt.imshow( Ic, extent=[np.min(x), np.max(x), np.min(y), np.max(y)] )
#     #cbar = fig.colorbar(im, ax=plt.gca(), pad=0.01)
#     #cbar.set_label(r'$\Delta$ Intensity', fontsize=15, labelpad=10)

#     #plt.scatter(pixel_coord_list[:, 0], pixel_coord_list[:, 1], color='blue', marker='.', label = 'DM actuators')
#     #plt.show() 

#     #plt.figure(); plt.imshow( I0-N0 ); plt.show()
#     #plt.figure(); plt.imshow( Ic-N0 ); plt.show()




#     # PSEUDO CODE 
#     # INIT ZWFS, CALIBRATION FIELD
    
#     # GET IM
#     # BUILD CONTROLLER
    
#     # 
#     # GET SIGNAL (FUNCTION)
#     #   

