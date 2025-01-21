# BaldrApp

Simulating Baldr - the Zernike Wavefront Sensor for Asgard

Includes 
- a  **PyQt** app for end-to-end simulatations and visualization of  Baldr operations (closed and open loop). Try:
    >>python apps/baldr_closed_loop_app/closed_loop_pyqtgraph.py

                                                    
- a **Streamlit** application that simulates a Zernike Wavefront Sensor optical system using Fresnel diffraction propagation to model system mis-alignments. The default setup is for simulating the last (critical) part of the optical train of Baldr. Try: 
    >>streamlit run apps/baldr_alignment_app/Baldr_Fresnel_App.py

- general packaged tools for simulating ZWFS's. These build upon the python pyZELDA package.


## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/courtney-barrer/BaldrApp

Depends on a forked version of the pyZELDA package (https://github.com/courtney-barrer/pyZELDA) which is specified in the requirements.txt.

if there are path issues after setup try install directly:

pip install pyzelda@git+https://github.com/courtney-barrer/pyZELDA.git@b42aaea5c8a47026783a15391df5e058360ea15e



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

