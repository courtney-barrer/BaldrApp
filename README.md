# BaldrApp

Simulating Baldr - the Zernike Wavefront Sensor (ZWFS) for VLTI/Asgard

Various modules and examples for testing end-to-end a ZWFS. Optionally includes, and extends the machinary of pyZelda to deal with specific details of Baldr, including its unique optics, coldstops, DMs and phasemasks. 
```
python
import baldrapp
```
## Installation
```
pip install baldrapp
```
This has a dependancy on a forked version of the pyZELDA package (https://github.com/courtney-barrer/pyZELDA) which must be installed seperately
```
pip install pyzelda@git+https://github.com/courtney-barrer/pyZELDA.git@b42aaea5c8a47026783a15391df5e058360ea15e
```    
Alternatvely the project can be cloned or forked from the Github:
```bash
git clone https://github.com/courtney-barrer/BaldrApp
```
The pip installation was tested on only on python 3.12.7. 

Older versions of the app also included:
- A  **PyQt** application for end-to-end simulatations and visualization of  Baldr operations (closed and open loop for a single telescope). The gui allows downloading of configuration files and telemetry. After pip installation try type in a terminal (warning: it takes 1-2 minutes to calibrate before the app will appear):
```
python -m baldrapp.apps.baldr_closed_loop_app.closed_loop_pyqtgraph
```
The app contains a command prompt that is exposed to the full python environment of the simulation. The default initialised mode is open loop with a weak rolling Kolmogorov atmosphere, and calibrated zonal matricies with zero gain. Some basic commands to test : 
```
zwfs_ns.ctrl.HO_ctrl.ki += 0.4 # put some non-zero gains

dynamic_opd_input=False #turn off rolling atmosphere phasescreen

M2C_0 = DM_basis.construct_command_basis( basis= "Zernike", 
number_of_modes = 20, without_piston=True).T # build a DM basis

dm_disturbance = M2C_0[5]* 1e-1 #put a static disturbance on the DM
```                                               
- A **Streamlit** application that simulates a Zernike Wavefront Sensor optical system using Fresnel diffraction propagation to model system mis-alignments. The default setup is for simulating the last (critical) part of the optical train of Baldr. After pip installation try type in a terminal: 
```
python -m baldrapp.apps.baldr_alignment_app.Baldr_Fresnel_App
```
These have not been upgraded for recent versions of BaldrApp so may not run (yet). 

