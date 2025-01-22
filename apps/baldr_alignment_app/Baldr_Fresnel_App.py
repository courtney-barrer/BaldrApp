#!/usr/bin/env python3
import streamlit as st
from streamlit.web import cli as stcli
from streamlit.runtime.scriptrunner import get_script_run_ctx

import numpy as np
import poppy
import matplotlib.pyplot as plt
from poppy.poppy_core import PlaneType
from poppy.zernike import zernike
import astropy.units as u
import os 
import sys
st.write("Starting app...")

# Thin lens functions and system
class lens:
    def __init__(self, x, f):
        self.f = f
        self.x = x
        
    def get_image(self, _object):
        do = (self.x - _object.x)
        if do != self.f:
            di = do * self.f / (do - self.f)
        else:
            di = np.inf
        
        M = -di/do
        i = obj(x=self.x + di, h=M * _object.h)
        return i

class obj:
    def __init__(self, x, h):
        self.x = x
        self.h = h

def propagate_object(lens_list, _object):
    object_list = [_object]
    for l in lens_list:
        next_object = l.get_image(object_list[-1])
        object_list.append(next_object)
    return object_list

# Function to bin the intensity image
def bin_image(image, bin_size):
    shape = (image.shape[0] // bin_size, bin_size,
             image.shape[1] // bin_size, bin_size)
    binned_image = image.reshape(shape).mean(-1).mean(1)
    return binned_image

# Define a custom optical element to apply Zernike aberration
class ZernikeAberration(poppy.AnalyticOpticalElement):
    def __init__(self, phase_aberration, name="Zernike Aberration"):
        super().__init__(name=name, planetype=poppy.poppy_core.PlaneType.pupil)
        self.phase_aberration = phase_aberration

    def get_opd(self, wave):
        return self.phase_aberration / (2 * np.pi / wave.wavelength.to(u.m).value)

# Phase Mask element
class PhaseMask(poppy.AnalyticOpticalElement):
    def __init__(self, D_phasemask, theta, T_on, T_off, name='Phase Mask'):
        super().__init__(name=name, planetype=PlaneType.pupil)
        self.D_phasemask = D_phasemask
        self.theta = theta
        self.T_on = T_on
        self.T_off = T_off

    def get_transmission(self, wave):
        y, x = wave.coordinates()  # 2D coordinate grid
        r = np.sqrt(x**2 + y**2)
        phase_mask = np.ones_like(r, dtype=complex)
        inside_mask = r <= (self.D_phasemask / 2)
        outside_mask = r > (self.D_phasemask / 2)
        phase_mask[inside_mask] = self.T_on * np.exp(1j * np.deg2rad(self.theta))
        phase_mask[outside_mask] = self.T_off
        return phase_mask

# Sidebar for user inputs
st.sidebar.title("Zernike Wavefront Sensor Inputs")

st.write("Instructions: Adjust alignment parameters in the side panel and press 'Update'. No image will be shown until you press update") # for logging

st.write("Reading in user variables.. Don't forget to press update on the side panel!") # for logging

# Initialize session_state
if "submitted" not in st.session_state:
    st.session_state.submitted = False
    
# Submit button
# Update button
if st.sidebar.button("Update"):
    st.write("changing session state..")
    st.session_state.submitted = True

# Input field inputs
wavelength = st.sidebar.number_input("Wavelength (um)", value=1.25)
zernike_mode = st.sidebar.number_input("Zernike Mode", value=None, step=1)
zernike_coefficient = st.sidebar.number_input("Zernike Coefficient", value=0.08)

# Phasemask inputs
phasemask_diameter = st.sidebar.number_input("Phasemask Diameter (lambda/D)", value=1.06)
theta = st.sidebar.number_input("Phase Mask Shift (degrees)", value=90)
T_on = st.sidebar.number_input("Phasemask On-Axis Transmission", value=1.0)
T_off = st.sidebar.number_input("Phasemask Off-Axis Transmission", value=1.0)

# Offset inputs
phasemask_offset = 1e-6 * st.sidebar.number_input("Phasemask Offset (um)", value=0.0)
lens1_offset = 1e-6 * st.sidebar.number_input("Collimating Lens Offset (um)", value=0.0)
lens2_offset = 1e-6 * st.sidebar.number_input("Pupil Imaging Lens Offset (um)", value=0.0)
coldstop_offset = 1e-6 * st.sidebar.number_input("Cold Stop Offset (um)", value=0.0)
detector_offset = 1e-6 * st.sidebar.number_input("Detector Offset (um)", value=0.0)

# offset_input_dict = {
#     "Phasemask Offset":phasemask_offset,
#     "Collimating Lens Offset":lens1_offset,
#     "Pupil Imaging Lens Offset":lens2_offset,
#     "Cold Stop Offset":coldstop_offset,
#     "Detector Offset":detector_offset
# }

# Include elements toggle switches
include_phase_mask = st.sidebar.checkbox("Include Phase Mask", value=True)
include_cold_stop = st.sidebar.checkbox("Include Cold Stop", value=True)



# Only generate plot if Update button is clicked
if st.session_state.submitted:
    st.write("updating to plot output intensity (after submit)..")
    # Hardcoded values
    npix = 256
    D = 12e-3  # Pupil diameter in meters
    f_oap = 254e-3  # OAP mirror focal length in meters
    f_lens1 = 30e-3  # Focal length of lens 1
    f_lens2 = 200e-3  # Focal length of lens 2
    cold_stop_diameter = 2.15e-3  # 2.15 mm
    distance_l1_to_l2 = 1228.570e-3

    # Thin lens system calculations for element positions
    x0 = -1  # Object position
    l1 = lens(x=0, f=f_oap)
    l2 = lens(x=f_oap + f_lens1, f=f_lens1)
    l3 = lens(x=f_oap + f_lens1 + distance_l1_to_l2, f=f_lens2)
    sys = [l1, l2, l3]

    oPup = obj(x=x0, h=D/2)
    pup_ims = propagate_object(sys, oPup)
    z_detector = pup_ims[-1].x

    # Absolute positions of the elements
    z_oap = 0
    z_fpm = f_oap
    z_lens1 = f_oap + f_lens1
    z_lens2 = f_oap + f_lens1 + distance_l1_to_l2
    z_coldstop = f_oap + f_lens1 + distance_l1_to_l2 + f_lens2
    abs_pos = np.array([0, z_fpm, z_lens1, z_lens2, z_coldstop, z_detector])
    
    # Calculated variables
    wavelength = wavelength * 1e-6 # convert um to m 
    D_phasemask = phasemask_diameter * 1.22 * wavelength * f_oap / D

    # Create optical elements
    oap_mirror = poppy.QuadraticLens(f_lens=f_oap * u.m, name='OAP Mirror')
    phase_mask = PhaseMask(D_phasemask=D_phasemask, theta=theta, T_on=T_on, T_off=T_off)
    lens1 = poppy.QuadraticLens(f_lens=f_lens1 * u.m, name='Lens 1')
    lens2 = poppy.QuadraticLens(f_lens=f_lens2 * u.m, name='Lens 2')
    cold_stop = poppy.CircularAperture(radius=cold_stop_diameter / 2 * u.m, name='Cold Stop')



    element_list = [oap_mirror, phase_mask, lens1, lens2, cold_stop] # we need to also explicitly define the full element_list for later
    # usr_element_list is primarily for creating usr_include_list which we also need later
    if (include_cold_stop) and (not include_phase_mask):
        usr_element_list = [oap_mirror, lens1, lens2, cold_stop]
    elif (not include_cold_stop) and (include_phase_mask):
        usr_element_list = [oap_mirror, phase_mask, lens1, lens2]
    elif (not include_cold_stop) and (not include_phase_mask):
        usr_element_list = [oap_mirror, lens1, lens2]
    else:
        usr_element_list = [oap_mirror, phase_mask, lens1, lens2, cold_stop]

    usr_include_list = [e.name for e in usr_element_list] 
    offset_list = [phasemask_offset, lens1_offset,lens2_offset, coldstop_offset, detector_offset] # MUST BE SAME ORDER AS element_list (without user removal!)
    

    abs_pos_with_offset = abs_pos.copy()
    for i, offset in enumerate( offset_list ):
        abs_pos_with_offset[i+1] += offset # i+1 since offset list does not include an offset for the first element (oap)
        
    rel_pos_with_offsets = np.diff( abs_pos_with_offset )

    wf = poppy.FresnelWavefront(beam_radius=D/2 *u.m, wavelength=wavelength *u.m, npix=npix, oversample=4)
    wf *= poppy.CircularAperture(radius=D/2) 

    # Apply Zernike aberration
    if zernike_mode is not None:
        y, x = wf.coordinates()
        r = np.sqrt(x**2 + y**2) / D * 2
        theta_vals = np.arctan2(y, x)
        n, m = poppy.zernike.noll_indices(zernike_mode)
        zernike_phase = zernike(n=n, m=m, rho=r, theta=theta_vals, outside=0)
        aberration_phase = 2 * np.pi * zernike_coefficient * zernike_phase
        zernike_aberration_element = ZernikeAberration(aberration_phase)
        wf *= zernike_aberration_element


    wf.propagate_fresnel( x0 * u.m ) 
    for element, rel_dist in zip( element_list ,   rel_pos_with_offsets ):
        # since we deal with relative positions we MUST go through all the elements (hence why we kept element_list above despite usr input)
        # if the user did not want to include one we still propagate the relative distance, we just 
        # don't aspply the elements transformation.
        if element.name in usr_include_list: # sometimes we might want to exclude an element (like the phasemask, or cold stop to analyse its effect)
            print( element.name )
            wf *= element
            
        wf.propagate_fresnel( rel_dist * u.m )
        
    # Get intensity at the detector
    intensity = bin_image(abs(wf.amplitude)**2, bin_size=16)

    st.write("Plotting..")
    # Display intensity image
    plt.figure()
    plt.imshow(intensity, cmap='inferno')
    plt.colorbar(label='Intensity')
    plt.title("Intensity at the Detector")
    st.pyplot(plt)



def main():
    # Check if the script is already running in Streamlit
    if get_script_run_ctx() is not None:
        # Your Streamlit app code goes here
        import streamlit as st
        st.title("Baldr Fresnel App")
        st.write("Welcome to the Baldr Fresnel App!")
    else:
        # Launch the app using streamlit run
        app_path = os.path.abspath(__file__)
        sys.argv = ["streamlit", "run", app_path]
        stcli.main()

if __name__ == "__main__":
    main()