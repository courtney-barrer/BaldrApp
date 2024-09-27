# BaldrApp

This is a **Streamlit** application that simulates a Zernike Wavefront Sensor optical system using Fresnel diffraction propagation.
The default setup is for simulating the last (critical) part of the optical train of Baldr - A Zernike Wavefront Sensor for the VLTI Asgard instrument suite.

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/.../BaldrApp.git


## Features
Key Features:
1. User Inputs via Sidebar:
    - Wavelength: Controls the light wavelength in micrometers.
    - Zernike Aberration: Users define Zernike mode and coefficient to simulate optical aberrations.
    - Phasemask Properties: Diameter, phase shift, on-axis/off-axis transmission coefficients.
    - Optical Element Offsets: Users can shift positions of elements like phase mask, lenses, and cold stop.
    - Element Inclusion: Toggle the inclusion/exclusion of key components (phase mask, cold stop).

2. System Propagation and Plot:
    - Calculates wavefront propagation through a multi-element optical system (lenses, phase mask, cold stop).
    - Fresnel diffraction propagation is used to model the interaction of light with optical elements.
    - Applies Zernike aberrations to simulate their effect on intensity measured at the detector.
    - Visualizes the intensity distribution at the detector using a heatmap.

3. Update Button:
    - Plot updates only when the "Update" button is pressed, making the app more efficient.
    - Input values are stored in `st.session_state` to prevent unnecessary re-runs.

4. Plotting the Results:
    - The app bins and displays the resulting intensity distribution at the detector.
    - A heatmap shows how system parameters affect the output.

