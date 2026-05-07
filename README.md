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

## VLTI/Baldr Simulator
BaldrApp comes with a simulator module that specifically reads the DM shared memory, updates the optical propagation of the beams based on the DM shared memory and ZWFS state, and then writes the simulated camera images to shared memory - in the exact same architecture as Baldr on the VLTI system. Therefore any RTC that consumes the camera frames from shared memory and writes to the DM shared memory can be plugged in and tested on this simulator in an agnostic fashion.


## VLTI/Baldr Simulator Architecture

The BaldrApp comes with a module designed to replicate the full 4 beam control data flow of the Baldr instrument on the VLTI system. It operates via a shared memory loop:
- Input: The module monitors and reads the DM shared memory for updated mirror commands.
- Optical Model: It calculates the optical propagation through the ZWFS based on the current state of the DM, phase mask and optics.
- Output: The resulting simulated camera frames are written directly to the camera shared memory.

Because this replicates Baldrs software interfaces, any RTC that consumes camera frames and writes DM commands via shared memory can interface with the simulator without modification. This allows for agnostic testing of the control loop against the simulated instrument.

To run VLTI Asgard/Baldr simulator (with shared memory - tested on Ubuntu 20.04). Install requirements in virtual enviornment. Once activated e.g.
```
source venv/bin/activate
```
then 
```
./baldrapp/apps/paranal_simulator/heimbal_simulation_servers.sh start 
```
This will create the camera and DM shared memory objects, begin a simulated camera and DM server, and run the Baldr simulator. The Paranal DM gui and camera shared memory viewer should open automatically and show updating frames. To stop all the server processes and close the simulator nicely, simply run 
```
./baldrapp/apps/paranal_simulator/heimbal_simulation_servers.sh stop
```
You can also view the status of the servers processes:
```
./baldrapp/apps/paranal_simulator/heimbal_simulation_servers.sh status 
```
If you are running this as another user on your linux you might need to give yourself permission to read/write to shared memory which can be done by 
```
sudo chown <user>:<user> /dev/shm/*some_shms* 
```
obviously using your username and whatever shared memory addresses (SHMs) you need to use

<!-- 
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
 -->

