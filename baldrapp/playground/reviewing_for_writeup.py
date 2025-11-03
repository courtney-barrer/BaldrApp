
import numpy as np
import matplotlib.pyplot as plt 
from scipy.integrate import quad
from scipy.interpolate import interp1d
import pandas as pd
from mpl_toolkits.axes_grid1 import make_axes_locatable
import math 
from matplotlib.widgets import Slider

import aotools

### Function from asgard-alignment/pyBaldr/utilities/

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




def get_phasemask_phaseshift( wvl, depth, dot_material = 'N_1405' ):
    """
    wvl is wavelength in micrometers
    depth is the physical depth of the phasemask in micrometers
    dot material is the material of phaseshifting object

    it is assumed phasemask is in air (n=1).
    N_1405 is photoresist used for making phasedots in Sydney
    """
    print( 'reminder wvl input should be um!')
    if dot_material == 'N_1405':
        # wavelengths in csv file are in nanometers
        pth = "/Users/bencb/Documents/ASGARD/BaldrApp/baldrapp/data/Exposed_Ma-N_1405_optical_constants.txt"
        df = pd.read_csv(pth, sep='\s+', header=1)
        f = interp1d(df['Wavelength(nm)'], df['n'], kind='linear',fill_value=np.nan, bounds_error=False)
        n = f( wvl * 1e3 ) # convert input wavelength um - > nm
        phaseshift = 2 * np.pi/ wvl  * depth * (n -1)
        return( phaseshift )
    
    else:
        raise TypeError('No corresponding dot material for given input. Try N_1405.')

# Planck's law function for spectral radiance
def planck_law(wavelength, T):
    """Returns spectral radiance (Planck's law) at a given wavelength and temperature."""
    h = 6.62607015e-34
    c = 299792458.0
    k = 1.380649e-23
    return (2 * h * c**2) / (wavelength**5) / (np.exp(h * c / (wavelength * k * T)) - 1)

# Function to find the weighted average wavelength (central wavelength)
def find_central_wavelength(lambda_cut_on, lambda_cut_off, T):
    # Define integrands for energy and weighted wavelength
    def _integrand_energy(wavelength):
        return planck_law(wavelength, T)

    def _integrand_weighted(wavelength):
        return planck_law(wavelength, T) * wavelength

    # Integrate to find total energy and weighted energy
    total_energy, _ = quad(_integrand_energy, lambda_cut_on, lambda_cut_off)
    weighted_energy, _ = quad(_integrand_weighted, lambda_cut_on, lambda_cut_off)
    
    # Calculate the central wavelength as the weighted average wavelength
    central_wavelength = weighted_energy / total_energy
    return central_wavelength


def get_theoretical_reference_pupils_with_aber( wavelength = 1.65e-6 ,F_number = 21.2, mask_diam = 1.2, coldstop_diam=None, coldstop_misalign=None, eta=0, phi= None, amp=None, diameter_in_angular_units = True, get_individual_terms=False, phaseshift = np.pi/2 , padding_factor = 4, debug= True, analytic_solution = True ) :
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
    coldstop_diam : diameter in lambda / D of focal plane coldstop
    coldstop_misalign : alignment offset of the cold stop (in units of image plane pixels)  
    phi : input phase aberrations (None by default). should be same size as pupil which by default is 2D grid of 2**9+1
    amp : field amplitude (optional), otherwise amp is 1 with pupil shape. The amp can be over the full dimension, it gets masked by the pupil internally 
    eta : ratio of secondary obstruction radius (r_2/r_1), where r2 is secondary, r1 is primary. 0 meams no secondary obstruction
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

    if (phi is None) and (amp is None):
        N = 2**9+1  # for parity (to not introduce tilt) works better ODD!  # Number of grid points (assumed to be square)
    elif (phi is not None) and (amp is None):
        N = phi.shape[0]
    elif (phi is None) and (amp is not None):
        N = amp.shape[0]
    else:
        assert phi.shape[0] == amp.shape[0]
        N = amp.shape[0]

    L_pupil = 2 * pupil_radius  # Pupil plane size (physical dimension)
    dx_pupil = L_pupil / N  # Sampling interval in the pupil plane
    x_pupil = np.linspace(-L_pupil/2, L_pupil/2, N)   # Pupil plane coordinates
    y_pupil = np.linspace(-L_pupil/2, L_pupil/2, N) 
    X_pupil, Y_pupil = np.meshgrid(x_pupil, y_pupil)
    
    


    # Define a circular pupil function
    pupil = (np.sqrt(X_pupil**2 + Y_pupil**2) > eta*pupil_radius) & (np.sqrt(X_pupil**2 + Y_pupil**2) <= pupil_radius)
    pupil = pupil.astype( complex )
    if phi is not None:
        pupil *= np.exp(1j * phi)
    else:
        phi = np.zeros( pupil.shape ) # added aberrations 
        
    if amp is not None:
        pupil *= amp

    # Zero padding to increase resolution
    # Increase the array size by padding (e.g., 4x original size)
    N_padded = N * padding_factor
    if (N % 2) != (N_padded % 2):  
        N_padded += 1  # Adjust to maintain parity
        
    pupil_padded = np.zeros((N_padded, N_padded)).astype(complex)
    #start_idx = (N_padded - N) // 2
    #pupil_padded[start_idx:start_idx+N, start_idx:start_idx+N] = pupil

    start_idx_x = (N_padded - N) // 2
    start_idx_y = (N_padded - N) // 2  # Explicitly ensure symmetry

    pupil_padded[start_idx_y:start_idx_y+N, start_idx_x:start_idx_x+N] = pupil


    phi_padded = np.zeros((N_padded, N_padded), dtype=float)
    phi_padded[start_idx_y:start_idx_y+N, start_idx_x:start_idx_x+N] = phi

    # Perform the Fourier transform on the padded array (normalizing for the FFT)
    #pupil_ft = np.fft.fftshift(np.fft.fft2(np.fft.ifftshift(pupil_padded))) # we do this laters
    
    # Compute the Airy disk scaling factor (1.22 * lambda * F)
    airy_scale = 1.22 * wavelength * F_number

    # Image plane sampling interval (adjusted for padding)
    #L_image = wavelength * F_number / dx_pupil  # Total size in the image plane
    #dx_image_padded = L_image / N_padded  # Sampling interval in the image plane with padding
    
    dx_image_padded = wavelength * F_number * (N / N_padded)
    L_image = dx_image_padded * N_padded

    if diameter_in_angular_units:
        x_image_padded = np.linspace(-L_image/2, L_image/2, N_padded) / airy_scale  # Image plane coordinates in Airy units
        y_image_padded = np.linspace(-L_image/2, L_image/2, N_padded) / airy_scale
    else:
        x_image_padded = np.linspace(-L_image/2, L_image/2, N_padded)  # Image plane coordinates in Airy units
        y_image_padded = np.linspace(-L_image/2, L_image/2, N_padded) 
        
    X_image_padded, Y_image_padded = np.meshgrid(x_image_padded, y_image_padded)

    if diameter_in_angular_units:
        mask = np.sqrt(X_image_padded**2 + Y_image_padded**2) <= mask_diam / 2 #4
    else: 
        mask = np.sqrt(X_image_padded**2 + Y_image_padded**2) <= mask_diam / 2 #4


    # --- convert misalignment from wvl/D to your image-plane units ---
    # ---- cold stop offset: wvl/D -> grid units ----
    if coldstop_misalign is not None:
        dx_wvld, dy_wvld = coldstop_misalign
    else:
        dx_wvld, dy_wvld = [0.0, 0.0]

    if diameter_in_angular_units:
        wvld_to_units = 1.0/1.22            # Airy radii per (wvl/D)
    else:
        wvld_to_units = F_number * wavelength  # meters per (wvl/D)
    dx_units = dx_wvld * wvld_to_units
    dy_units = dy_wvld * wvld_to_units

    if coldstop_diam is not None:
        if diameter_in_angular_units:
            cs_radius_units = (coldstop_diam * (1.0/1.22)) / 2.0
        else:
            cs_radius_units = (coldstop_diam * (F_number * wavelength)) / 2.0
        coldmask = (np.hypot(X_image_padded - dx_units, Y_image_padded - dy_units) <= cs_radius_units).astype(float)
    else:
        coldmask = np.ones_like(X_image_padded)


    # if coldstop_misalign is not None:
    #     dx_wvld, dy_wvld = coldstop_misalign
    # else:
    #     dx_wvld, dy_wvld = [0,0]
    

    # if diameter_in_angular_units:
    #     # Your X_image_padded, Y_image_padded are in "Airy radii" units set by:
    #     # airy_scale = 1.22 * wavelength * F_number
    #     # 1 (wvl/D) equals (F_number * wavelength) in meters,
    #     # which is (1 / 1.22) Airy radii on this normalized grid.
    #     wvld_to_units = 1.0 / 1.22                      # Airy radii per (wvl/D)
    #     dx_units = dx_wvld * wvld_to_units
    #     dy_units = dy_wvld * wvld_to_units
    # else:
    #     # Your X_image_padded, Y_image_padded are in meters.
    #     # 1 (wvl/D) = F_number * wavelength  [meters]
    #     wvld_to_units = F_number * wavelength           # meters per (wvl/D)
    #     dx_units = dx_wvld * wvld_to_units
    #     dy_units = dy_wvld * wvld_to_units
        
    # # if coldstop_diam is not None:
    # #     coldmask = np.sqrt(X_image_padded**2 + Y_image_padded**2) <= coldstop_diam / 4
    # # else:
    # #     coldmask = np.ones(X_image_padded.shape)
    # if coldstop_diam is not None: # apply also the cold stop offset 
    #     coldmask = np.sqrt((X_image_padded-dx_units)**2 + (Y_image_padded-dy_units)**2) <= coldstop_diam / 2 #4
    # else:
    #     coldmask = np.ones(X_image_padded.shape)

    pupil_ft = np.fft.fft2(np.fft.ifftshift(pupil_padded))  # Remove outer fftshift
    pupil_ft = np.fft.fftshift(pupil_ft)  # Shift only once at the end

    psi_B = coldmask * pupil_ft
                            
    b = np.fft.fftshift( np.fft.ifft2( mask * psi_B ) ) # we do mask here because really the cold stop is after phase mask in physical system

    
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
        

        # out formula ----------
        #if measured_pupil!=None:
        #    P = measured_pupil / np.mean( P[P > np.mean(P)] ) # normalize by average value in Pupil
        P = np.abs(pupil_padded).real  # we already dealt with the complex part in this analytic expression which is in phi
        Ic = ( P**2 + abs(M)**2 + 2* P* abs(M) * np.cos(phi_padded + mu) ) #+ beta)
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
    
        return( P, Ic )




################ FOR TESTING SCINTILLATION 

def azimuthal_average(psd2d, fx, fy, nbins=None):
    FX, FY = np.meshgrid(fx, fy, indexing='xy')
    fr = np.sqrt(FX**2 + FY**2)
    fmax = fr.max()
    if nbins is None:
        nbins = min(200, psd2d.shape[0] // 2)
    edges = np.linspace(0.0, fmax, nbins + 1)
    centers = 0.5*(edges[1:] + edges[:-1])
    idx = np.digitize(fr.ravel(), edges) - 1
    valid = (idx >= 0) & (idx < nbins)
    sums = np.bincount(idx[valid], weights=psd2d.ravel()[valid], minlength=nbins)
    cnts = np.bincount(idx[valid], minlength=nbins).astype(float)
    psd1d = sums / np.maximum(cnts, 1.0)
    return centers, psd1d

def periodogram2d(img, dx, use_hann=True):
    """
    2D periodogram with physical frequency axes (cycles/m).
    Normalized so that sum(PSD)*dfx*dfy ≈ mean(img_windowed^2).
    """
    if use_hann:
        win = np.hanning(img.shape[0])
        W = np.outer(win, win)
        imgw = img * W
        norm = (W**2).sum()
    else:
        imgw = img
        norm = img.size
    F = np.fft.fft2(imgw)
    fx = np.fft.fftfreq(img.shape[1], d=dx)  # cycles/m
    fy = np.fft.fftfreq(img.shape[0], d=dx)
    P2D = (np.abs(F)**2) * (dx**2) / norm     # units: img^2 * m^2
    return fx, fy, P2D

def check_scintillation_psd(U, wavelength, z, dx, r0, L0, ffit=(0.5, 5.0), title_suffix=""):
    """
    Compare measured PSD of δI with short-exposure theory.
    """
    I = np.abs(U)**2
    dI = I / I.mean() - 1.0

    fx, fy, P2D = periodogram2d(dI, dx=dx, use_hann=True)
    f, P1D = azimuthal_average(P2D, fx, fy)

    # theory up to a scale factor
    fresnel = 4.0 * np.sin(np.pi * wavelength * z * (f**2))**2
    Wphi = 0.0029 * r0**(-5/3.0) * (f**2 + (1.0/L0)**2)**(-11/6.0)
    W_I_th = Wphi * fresnel

    # scale theory to measured mid-band for overlay
    band = (f > 0) & (np.isfinite(P1D)) & (P1D > 0) & (f >= ffit[0]) & (f <= ffit[1])
    scale = np.median(P1D[band]) / np.median(W_I_th[band]) if band.sum() > 10 else 1.0
    W_I_th_scaled = 5 * scale * W_I_th

    # fit slope on measured PSD in band
    x = np.log10(f[band]); y = np.log10(P1D[band])
    slope = np.polyfit(x, y, 1)[0] if band.sum() > 5 else np.nan

    # plot
    msk = (f > 0) & np.isfinite(P1D) & (P1D > 0)
    plt.figure(figsize=(6.4, 4.4))
    plt.loglog(f[msk], P1D[msk], label='measured PSD of $\delta I$')
    plt.loglog(f[msk], W_I_th_scaled[msk], '--', label='theory (scaled)')
    fidx = np.argmin( abs( np.median( np.diff(P1D) ) - np.diff(P1D) ) )
    plt.loglog(f[msk], P1D[msk][fidx]/f[msk][fidx] * f[msk]**(-11/3), '--', color='k',label=r'$f^{-11/3}$')
    #plt.axvspan(ffit[0], ffit[1], color='k', alpha=0.06, label='fit band')
    plt.xlabel('spatial frequency  $f$  [cycles/m]')
    plt.ylabel('PSD of $\\delta I$  []')
    #plt.title(f'scintillation PSD vs theory{title_suffix} (slope~{slope:.2f})')
    plt.grid(True, which='both', ls=':', alpha=0.4)
    plt.legend()
    plt.tight_layout()
    plt.show()

    print(f"Fitted log–log slope in {ffit} cycles/m band: {slope:.3f}")
    return f, P1D, W_I_th_scaled, slope

def check_phase_psd(scrn, dx, L0, ffit=(0.5, 5.0), title='phase screen PSD (expect -11/3 slope mid-band)'):
    fx, fy, P2D = periodogram2d(scrn, dx=dx, use_hann=True)
    f, P1D = azimuthal_average(P2D, fx, fy)
    # fit slope in band away from DC and Nyquist; avoid outer-scale roll-off region (f < 1/L0)
    band = (f > max(1.0/L0, ffit[0])) & (f <= ffit[1]) & (P1D > 0) & np.isfinite(P1D)
    slope = np.polyfit(np.log10(f[band]), np.log10(P1D[band]), 1)[0] if band.sum() > 5 else np.nan

    msk = (f > 0) & (P1D > 0) & np.isfinite(P1D)
    plt.figure(figsize=(6.4, 4.4))
    plt.loglog(f[msk], P1D[msk], label='measured PSD $W_\\phi(f)$')
    # reference -11/3 line
    if np.any(msk):
        f_ref = np.array([max(1.0/L0, ffit[0]), ffit[1]])
        y_mid = np.exp(np.interp(np.log(np.sqrt(f_ref.prod())), np.log(f[msk]), np.log(P1D[msk])))
        ref = y_mid * (f_ref / np.sqrt(f_ref.prod()))**(-11/3)
        plt.loglog(f_ref, ref, ':', label='ref. slope $-11/3$')
    plt.axvline(1.0/L0, color='grey', lw=1, ls='--', label='$1/L_0$')
    plt.xlabel('spatial frequency  $f$  [cycles/m]')
    plt.ylabel('PSD of $\\phi$  [arb. units]')
    plt.title(f'{title}  (slope~{slope:.2f})')
    plt.grid(True, which='both', ls=':', alpha=0.4)
    plt.legend()
    plt.tight_layout()
    plt.show()
    print(f"Phase PSD fitted slope in {ffit} (≥1/L0) band: {slope:.3f}")
    return f, P1D, slope





phasemask_parameters = {  
                        "J5": {"depth":0.474 ,  "diameter":32},
                        "J4": {"depth":0.474 ,  "diameter":36}, 
                        "J3": {"depth":0.474 ,  "diameter":44}, 
                        "J2": {"depth":0.474 ,  "diameter":54},
                        "J1": {"depth":0.474 ,  "diameter":65},
                        "H1": {"depth":0.654 ,  "diameter":68},  
                        "H2": {"depth":0.654 ,  "diameter":53}, 
                        "H3": {"depth":0.654 ,  "diameter":44}, 
                        "H4": {"depth":0.654 ,  "diameter":37},
                        "H5": {"depth":0.654 ,  "diameter":31}
                        }



#### SCINTILLATION 
#following example from AOtools paper 
nx_size = 2**7 #128
D = 1.8 #8.0
pxl_scale =  D/nx_size
r0 = 0.164
L0  = 10
#stencil_length_factor = 2**7 #4 #32
phasescreen =  aotools.turbulence.infinitephasescreen.PhaseScreenVonKarman(nx_size,pxl_scale,r0,L0)
#aotools.turbulence.infinitephasescreen.PhaseScreenKolmogorov(nx_size,pxl_scale,r0,L0,stencil_length_factor)

wavefront = np.exp(1J *  phasescreen.scrn )

wavelength = 500e-9
propagation_distance = 10000.0
propagated_screen = aotools.opticalpropagation.angularSpectrum(inputComplexAmp=wavefront,
                                                               z=propagation_distance, 
                                                               wvl=wavelength, 
                                                               inputSpacing= pxl_scale, 
                                                               outputSpacing =pxl_scale
                                                               )

# nice_heatmap_subplots(im_list=[phasescreen.scrn,abs(propagated_screen)**2],title_list=['phasescreen','scintillation'])
# plt.show()


# ---------- run checks  ----------

# choose a mid-band for slope fit (tune to your dx and z):
# Nyquist is 1/(2*dx). keep away from DC and too close to Nyquist / Fresnel zeros.
f_nyq = 1.0/(2.0*pxl_scale )
ffit_scint = (0.5, min(5.0, 0.5*f_nyq))   # cycles/m; adjust if needed
ffit_phase = (0.5, min(5.0, 0.5*f_nyq))

# 1) short-exposure scintillation PSD check
_ = check_scintillation_psd(propagated_screen, wavelength, propagation_distance, pxl_scale , r0, L0, ffit=ffit_scint,
                            title_suffix=f' (D={D} m, N={nx_size}, dx={pxl_scale:.3e} m)')

# 2) phase-screen PSD check (expect -11/3 mid-band, roll-off below 1/L0)
_ = check_phase_psd(phasescreen.scrn, dx=dx, L0=L0, ffit=ffit_phase)



# animate it 
phase_screen_list = []
scint_screen_list = []
for _ in range(100):
    phasescreen.add_row()
    wavefront = np.exp(1J *  phasescreen.scrn )
    propagated_screen = aotools.opticalpropagation.angularSpectrum(inputComplexAmp=wavefront,
                                                               z=propagation_distance, 
                                                               wvl=wavelength, 
                                                               inputSpacing= pxl_scale, 
                                                               outputSpacing =pxl_scale
                                                               )
    phase_screen_list.append( phasescreen.scrn )
    scint_screen_list.append( propagated_screen )

# lets see it 
display_images_with_slider(image_lists = [phase_screen_list, [abs(ss)**2 for ss in scint_screen_list]], plot_titles=None)


def upsample_by_factor(ar: np.ndarray, f: int | tuple[int, int]) -> np.ndarray:
    """
    Upsample 2D array by integer factor(s) via block replication (nearest-neighbour).
    If f is an int, uses the same factor on both axes. If f=(fy, fx), uses per-axis factors.
    """
    ar = np.asarray(ar)
    if ar.ndim != 2:
        raise ValueError("ar must be 2D")
    fy, fx = (f, f) if isinstance(f, int) else f
    if fy < 1 or fx < 1:
        raise ValueError("factors must be positive integers")
    return np.repeat(np.repeat(ar, fy, axis=0), fx, axis=1)


def pad_to_shape_edge(arr: np.ndarray, target_shape: tuple[int, int]) -> np.ndarray:
    """
    Grow a 2D array to target_shape by copying its edges (last row/col).
    Equivalent to np.pad(..., mode='edge').
    """
    if arr.ndim != 2:
        raise ValueError("arr must be 2D")
    m, n = arr.shape
    M, N = target_shape
    if M < m or N < n:
        raise ValueError("target_shape must be >= current shape in both dims")

    pad_rows = M - m
    pad_cols = N - n
    return np.pad(arr, ((0, pad_rows), (0, pad_cols)), mode="edge")


def upsample( ar, target_size ):
    out_almost = upsample_by_factor(ar, target_size//len(ar) ) # 
    if np.mod( target_size, len(ar) ) != 0: # not even divisor, we just pad! 
        out = pad_to_shape_edge( out_almost , target_shape = (target_size,target_size))
    else :
        out = out_almost
    return( out )


amp = upsample(scint_screen_list[0], N )


#### 

#### 
N = 2**7 + 1
T = 1900 #K lab thermal source temperature 
lambda_cut_on, lambda_cut_off =  1.38, 1.82 # um
wvl = find_central_wavelength(lambda_cut_on, lambda_cut_off, T) # central wavelength of Nice setup
mask = "H3"
F_number = 21.2
coldstop_diam = 4.04 #according to calc in thesis 8.07 lmabda/D bright, 4.04 lambda/D faint
mask_diam = 1.22 * F_number * wvl / phasemask_parameters[mask]['diameter']
eta = 138 / 1800 #0.647/4.82 #~= 1.1/8.2 (i.e. UTs) # ratio of secondary obstruction (UTs), secondary obstruction ATs 138 mm / 1800mm, (https://www.eso.org/sci/facilities/paranal/telescopes/vlti/subsystems/at/technic.html)

###################
# TT, focus basis 

x = np.linspace(-1, 1, N)
X, Y = np.meshgrid(x, x, indexing='xy')
P = (X**2 + Y**2) <= 1  # circular pupil mask
tip  = X * P
tilt = Y * P
# Normalize each to 1 rad RMS within the pupil
tip  /= np.sqrt(np.mean(tip[P]**2))
tilt /= np.sqrt(np.mean(tilt[P]**2))

R2 = X**2 + Y**2
focus = (2.0*R2 - 1.0) * P          # Zernike-like defocus (∝ Z4)

# Remove tiny numerical piston/tilt leakage, then normalize to 1 rad RMS
focus -= focus[P].mean()
focus -= ((focus[P]*tip[P]).mean()  / (tip[P]**2).mean())  * tip
focus -= ((focus[P]*tilt[P]).mean() / (tilt[P]**2).mean()) * tilt
focus /= np.sqrt(np.mean(focus[P]**2))

basis = [tip,tilt,focus]
###################

# np.sum( tip**2*P ) / np.sum(P**2) == 1
P, Ic = get_theoretical_reference_pupils_with_aber( wavelength = wvl ,
                                              F_number = F_number , 
                                              mask_diam = mask_diam, 
                                              coldstop_diam=10, #coldstop_diam,
                                              coldstop_misalign = [1,0], #lambda/D units
                                              eta = eta, 
                                              amp = abs(amp),
                                              phi = -0.3 * basis[1], #0 * basis[1], #+ -0.3 * basis[2] ,
                                              diameter_in_angular_units = True, 
                                              get_individual_terms=False, 
                                              phaseshift = get_phasemask_phaseshift(wvl=wvl, depth = phasemask_parameters[mask]['depth'], dot_material='N_1405') , 
                                              padding_factor = 6, 
                                              debug= False, 
                                              analytic_solution = False )

nice_heatmap_subplots(im_list=[abs(P),abs(Ic)])
plt.show()


P_list = []
Ic_list = []
for it in range( 10 ):
    print(it)
    # Scintillation 
    print("prop scintillation screen")
    phasescreen.add_row()
    wavefront = np.exp(1J *  phasescreen.scrn )
    propagated_screen = aotools.opticalpropagation.angularSpectrum(inputComplexAmp=wavefront,
                                                               z=propagation_distance, 
                                                               wvl=wavelength, 
                                                               inputSpacing= pxl_scale, 
                                                               outputSpacing =pxl_scale
                                                               )
    print("upsample it scintillation screen")
    amp = upsample(propagated_screen, N )

    print("ZWFS propagation")
    P, Ic = get_theoretical_reference_pupils_with_aber( wavelength = wvl ,
                                              F_number = F_number , 
                                              mask_diam = mask_diam, 
                                              coldstop_diam=10, #coldstop_diam,
                                              coldstop_misalign = [0,0], #lambda/D units
                                              eta = eta, 
                                              amp = abs(amp),
                                              phi = 0 * basis[1], #0 * basis[1], #+ -0.3 * basis[2] ,
                                              diameter_in_angular_units = True, 
                                              get_individual_terms=False, 
                                              phaseshift = get_phasemask_phaseshift(wvl=wvl, depth = phasemask_parameters[mask]['depth'], dot_material='N_1405') , 
                                              padding_factor = 6, 
                                              debug= False, 
                                              analytic_solution = False )
    

    P_list.append(abs(P))
    Ic_list.append(abs(Ic))


def crop_imlist( l , f = 4):
    return [pp[len(pp)//2-len(pp)//f:len(pp)//2+len(pp)//f,len(pp)//2-len(pp)//f:len(pp)//2+len(pp)//f] for pp in l]

display_images_with_slider(image_lists = [crop_imlist( P_list , f = 4),crop_imlist( Ic_list, f = 4)], plot_titles=None)




#######################################


#### SCINTILLATION 
#following example from AOtools paper 
nx_size = 2**7 #128
D = 1.8 #8.0
pxl_scale =  D/nx_size
r0_scint = 0.164
L0_scint  = 10

r0 = (0.164) * (1.65/0.5)**(6/5)
L0  = 25
#stencil_length_factor = 2**7 #4 #32
scint_phasescreen =  aotools.turbulence.infinitephasescreen.PhaseScreenVonKarman(nx_size,pxl_scale,r0_scint,L0_scint)

phase_phasescreen =  aotools.turbulence.infinitephasescreen.PhaseScreenVonKarman(nx_size,pxl_scale,r0,L0)


def update_scintillation( high_alt_phasescreen , pxl_scale, wavelength, final_size = None,jumps = 1):
    for _ in range(jumps):
        high_alt_phasescreen.add_row()
    wavefront = np.exp(1J *  high_alt_phasescreen.scrn ) # amplitude mean ~ 1 
    propagated_screen = aotools.opticalpropagation.angularSpectrum(inputComplexAmp=wavefront,
                                                               z=propagation_distance, 
                                                               wvl=wavelength, 
                                                               inputSpacing = pxl_scale, 
                                                               outputSpacing = pxl_scale
                                                               )
    print("upsample it scintillation screen")
    if final_size is not None:
        amp = upsample(propagated_screen, final_size ) # This oversamples to nearest multiple size, and then pads the rest with repeated rows, not the most accurate but fastest. Negligible if +1 from even number
    else:
        amp = propagated_screen

    return( abs(amp).T ) # amplitude of field, not intensity (amp^2)! rotate 90 degrees so not correlated with phase 

def update_phase( phasescreen, final_size =None , jumps = 1):
    for _ in range(jumps-1):
        phasescreen.add_row()
    
    phase = upsample(phasescreen.add_row(), final_size  )
    return( phase )



def sum_subarrays(array, block_size):
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
    summed_subarrays = reshaped.sum(axis=(1, 3))
    
    return summed_subarrays


amp_list = []
phase_list = []
Ic_list = []
Icsub_list = []
mode = "bright"
for it in range(50):

    # roll scint scint_screen
    amp = update_scintillation( high_alt_phasescreen=scint_phasescreen , pxl_scale=pxl_scale , wavelength=wvl * 1e-6, final_size = N-1, jumps = 10)
    # roll phase screen 
    phase =  update_phase( phase_phasescreen, final_size = N-1 , jumps = 5) 

    #   first stage ao 
    # to do 

    #     baldr_dm 
    # to do 

    # propagate 
    P, Ic = get_theoretical_reference_pupils_with_aber( wavelength = wvl ,
                                              F_number = F_number , 
                                              mask_diam = mask_diam, 
                                              coldstop_diam=10, #coldstop_diam,
                                              coldstop_misalign = [0,0], #lambda/D units
                                              eta = eta, 
                                              amp = abs(amp),
                                              phi = phase, #0 * basis[1], #+ -0.3 * basis[2] ,
                                              diameter_in_angular_units = True, 
                                              get_individual_terms=False, 
                                              phaseshift = get_phasemask_phaseshift(wvl=wvl, depth = phasemask_parameters[mask]['depth'], dot_material='N_1405') , 
                                              padding_factor = 6, 
                                              debug= False, 
                                              analytic_solution = False )
    

    #     detect 
    # to do 
    if "bright" in mode:
        Ic_sub = sum_subarrays(abs(Ic), block_size=(8,8))
        
    elif "faint" in mode:
        Ic_sub = sum_subarrays(abs(Ic), block_size=(16,16))
        
    #     reco
    #     recon 
    # to do 

    #     update dm 
    # to do 

    #     save telemetry 
    # to do 
    amp_list.append(abs(P))
    phase_list.append( phase )
    Ic_list.append(abs(Ic))
    Icsub_list.append(Ic_sub)


display_images_with_slider(image_lists = [crop_imlist( amp_list , f = 4),phase_list, crop_imlist( Ic_list, f = 4), Icsub_list ], plot_titles=None)
