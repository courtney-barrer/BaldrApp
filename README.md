# BaldrApp

Simulating Baldr - the Zernike Wavefront Sensor for Asgard

Includes 
- A  **PyQt** app for end-to-end simulatations and visualization of  Baldr operations (closed and open loop for a single telescope). The gui allows downloading of configuration files and telemetry. After pip installation try type in a terminal prompt (warning: it takes 1-2 minutes to calibrate before the app will appear):
```
closed_loop_pyqtgraph.py
```                                                
- a **Streamlit** application that simulates a Zernike Wavefront Sensor optical system using Fresnel diffraction propagation to model system mis-alignments. The default setup is for simulating the last (critical) part of the optical train of Baldr. After pip installation try type in a terminal prompt: 
```
Baldr_Fresnel_App.py
```
- general packaged tools for simulating ZWFS's. These build upon the python pyZELDA package ().


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


