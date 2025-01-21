# BaldrApp

Simulating Baldr - the Zernike Wavefront Sensor for Asgard

Includes 
- A  **PyQt** app for end-to-end simulatations and visualization of  Baldr operations (closed and open loop). Try:
```
python apps/baldr_closed_loop_app/closed_loop_pyqtgraph.py
```
                                                    
- a **Streamlit** application that simulates a Zernike Wavefront Sensor optical system using Fresnel diffraction propagation to model system mis-alignments. The default setup is for simulating the last (critical) part of the optical train of Baldr. Try: 
```
streamlit run apps/baldr_alignment_app/Baldr_Fresnel_App.py
```
- general packaged tools for simulating ZWFS's. These build upon the python pyZELDA package ().


## Installation
    ```
    pip install baldrapp
    
This has a dependancy on a forked version of the pyZELDA package (https://github.com/courtney-barrer/pyZELDA) which must be installed seperately
    ```
    pip install pyzelda@git+https://github.com/courtney-barrer/pyZELDA.git@b42aaea5c8a47026783a15391df5e058360ea15e
    
Alternatvely the project can be cloned or forked from the Github:
   ```bash
   git clone https://github.com/courtney-barrer/BaldrApp



