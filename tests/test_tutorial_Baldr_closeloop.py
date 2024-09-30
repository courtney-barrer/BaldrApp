
import numpy as np
import matplotlib.pyplot as plt
from types import SimpleNamespace
import importlib 
from matplotlib.gridspec import GridSpec
from mpl_toolkits.axes_grid1 import make_axes_locatable

import Baldr_closeloop as bldr
import DM_basis as gen_basis
import utilities as util



################## TEST 0 
# configure our zwfs 
grid_dict = {
    "D":1, # diameter of beam 
    "N" : 64, # number of pixels across pupil diameter
    "padding_factor" : 4, # how many pupil diameters fit into grid x axis
    # TOTAL NUMBER OF PIXELS = padding_factor * N 
    }

optics_dict = {
    "wvl0" :1.65e-6, # central wavelength (m) 
    "F_number": 21.2, # F number on phasemask
    "mask_diam": 1.06, # diameter of phaseshifting region in diffraction limit units (physical unit is mask_diam * 1.22 * F_number * lambda)
    "theta": 1.57079, # phaseshift of phasemask 
}

dm_dict = {
    "dm_model":"BMC-multi-3.5",
    "actuator_coupling_factor":0.7,# std of in actuator spacing of gaussian IF applied to each actuator. (e.g actuator_coupling_factor = 1 implies std of poke is 1 actuator across.)
    "dm_pitch":1,
    "dm_aoi":0, # angle of incidence of light on DM 
    "opd_per_cmd" : 3e-6, # peak opd applied at center of actuator per command unit (normalized between 0-1) 
    "flat_rmse" : 20e-9 # std (m) of flatness across Flat DM  
    }

grid_ns = SimpleNamespace(**grid_dict)
optics_ns = SimpleNamespace(**optics_dict)
dm_ns = SimpleNamespace(**dm_dict)




################## TEST 1
# check dm registration on pupil (wavespace)
zwfs_ns = bldr.init_zwfs(grid_ns, optics_ns, dm_ns)

phi, phi_internal,  N0, I0, Intensity = bldr.test_propagation( zwfs_ns )

fig = plt.figure() 
im = plt.imshow( N0, extent=[np.min(zwfs_ns.grid.wave_coord.x), np.max(zwfs_ns.grid.wave_coord.x),\
    np.min(zwfs_ns.grid.wave_coord.y), np.max(zwfs_ns.grid.wave_coord.y)] )
cbar = fig.colorbar(im, ax=plt.gca(), pad=0.01)
cbar.set_label(r'Pupil Intensity', fontsize=15, labelpad=10)

plt.scatter(zwfs_ns.grid.dm_coord.act_x0_list_wavesp, zwfs_ns.grid.dm_coord.act_y0_list_wavesp, color='blue', marker='.', label = 'DM actuators')
plt.show() 

################## TEST 2
# check pupil intensities 
fig,ax = plt.subplots( 1,4 )
ax[0].imshow( util.get_DM_command_in_2D( zwfs_ns.dm.current_cmd ))
ax[0].set_title('dm cmd')
ax[1].set_title('OPD wavespace')
ax[1].imshow( phi )
ax[2].set_title('ZWFS Intensity')
ax[2].imshow( Intensity )
ax[3].set_title('ZWFS reference Intensity')
ax[3].imshow( I0 )


################## TEST 3 
# test updating the DM registration 
zwfs_ns = bldr.init_zwfs(grid_ns, optics_ns, dm_ns)

# redefining the affine transform between DM coordinates and the wavefront space
a, b, c, d = zwfs_ns.grid.D/np.ptp(zwfs_ns.grid.wave_coord.x)/2, 0, 0, grid_ns.D/np.ptp(zwfs_ns.grid.wave_coord.x)/2  # Parameters for affine transform (identity for simplicity)

# offset 5% of pupil 
t_x, t_y = np.mean(zwfs_ns.grid.wave_coord.x) + 0.05 * zwfs_ns.grid.D, np.mean(zwfs_ns.grid.wave_coord.x)  # Translation in phase space

# we could also introduce mis-registrations by rolling input pupil 
dm_act_2_wave_space_transform_matrix = np.array( [[a,b,t_x],[c,d,t_y]] )

zwfs_ns = bldr.update_dm_registration( dm_act_2_wave_space_transform_matrix , zwfs_ns )

opd_atm, opd_internal, opd_dm, phi, N0, I0,  Intensity  = bldr.test_propagation( zwfs_ns )


fig = plt.figure() 
im = plt.imshow( N0, extent=[np.min(zwfs_ns.grid.wave_coord.x), np.max(zwfs_ns.grid.wave_coord.x),\
    np.min(zwfs_ns.grid.wave_coord.y), np.max(zwfs_ns.grid.wave_coord.y)] )
cbar = fig.colorbar(im, ax=plt.gca(), pad=0.01)
cbar.set_label(r'Pupil Intensity', fontsize=15, labelpad=10)

plt.scatter(zwfs_ns.grid.dm_coord.act_x0_list_wavesp, zwfs_ns.grid.dm_coord.act_y0_list_wavesp, color='blue', marker='.', label = 'DM actuators')
plt.show() 



################## TEST 4
# test DM basis generation 
basis_name_list = ['Hadamard', "Zonal", "Zonal_pinned_edges", "Zernike", "Zernike_pinned_edges", "fourier", "fourier_pinned_edges"]

fig, ax = plt.subplots( len( basis_name_list), len( basis_name_list) ,figsize=(15,15) )
for i,b in enumerate( basis_name_list ) :
    
    print( b )
    basis_test = gen_basis.construct_command_basis( basis= b, number_of_modes = 20, Nx_act_DM = 12, Nx_act_basis = 12, act_offset=(0,0), without_piston=True)
    
    for m in range( ax.shape[1] ):
        ax[ i, m ].imshow( util.get_DM_command_in_2D( basis_test.T[m] ) )
    
    ax[i, 0].set_ylabel( b )
    #print( basis_test )
    
#plt.savefig( "/Users/bencb/Downloads/baldr_bases.png",dpi=300)
plt.show() 



################## TEST 5
# Get reference intensities 
zwfs_ns = bldr.init_zwfs(grid_ns, optics_ns, dm_ns)

opd_input = 0 * zwfs_ns.grid.pupil_mask *  np.random.randn( *zwfs_ns.grid.pupil_mask.shape)

opd_internal = 10e-9 * zwfs_ns.grid.pupil_mask * np.random.randn( *zwfs_ns.grid.pupil_mask.shape)

amp_input = 1e2 * zwfs_ns.grid.pupil_mask

I0 = bldr.get_I0(  opd_input  = opd_input ,   amp_input = amp_input,\
    opd_internal = opd_internal,  zwfs_ns= zwfs_ns , detector=None )

N0 = bldr.get_N0(  opd_input  = opd_input ,   amp_input = amp_input,\
    opd_internal = opd_internal,  zwfs_ns= zwfs_ns , detector=None )


################## TEST 6
# classify pupil regions and plot them
zwfs_ns = bldr.init_zwfs(grid_ns, optics_ns, dm_ns)

opd_input = 0 * zwfs_ns.grid.pupil_mask *  np.random.randn( *zwfs_ns.grid.pupil_mask.shape)

opd_internal = 10e-9 * zwfs_ns.grid.pupil_mask * np.random.randn( *zwfs_ns.grid.pupil_mask.shape)

amp_input = 1e2 * zwfs_ns.grid.pupil_mask

zwfs_ns = bldr.classify_pupil_regions( opd_input,  amp_input ,  opd_internal,  zwfs_ns , detector=None)

fig,ax = plt.subplots( 1, len(zwfs_ns.pupil_regions.__dict__ ) ,figsize=(10,10) )
for k,axx in zip( zwfs_ns.pupil_regions.__dict__,ax.reshape(-1)):
    axx.imshow(zwfs_ns.pupil_regions.__dict__[k] )
    axx.set_title( k ) 
plt.show()


################## TEST 7
# Build IM  and look at Eigenmodes! 
zwfs_ns = bldr.init_zwfs(grid_ns, optics_ns, dm_ns)

opd_input = 0 * zwfs_ns.grid.pupil_mask *  np.random.randn( *zwfs_ns.grid.pupil_mask.shape)

opd_internal = 10e-9 * zwfs_ns.grid.pupil_mask * np.random.randn( *zwfs_ns.grid.pupil_mask.shape)

amp_input = 1e4 * zwfs_ns.grid.pupil_mask

# we must first define our pupil regions before building 
zwfs_ns = bldr.classify_pupil_regions( opd_input,  amp_input ,  opd_internal,  zwfs_ns , detector=None)

basis_name_list = ['Hadamard', "Zonal", "Zonal_pinned_edges", "Zernike", "Zernike_pinned_edges", "fourier", "fourier_pinned_edges"]

# perfect field only with internal opd aberrations 
# different poke methods 
Nmodes = 100
basis = 'Zonal_pinned_edges'
M2C_0 = gen_basis.construct_command_basis( basis= basis, number_of_modes = Nmodes, without_piston=True).T  

IM_00 = bldr.build_IM( zwfs_ns ,  calibration_opd_input= 0 *zwfs_ns.grid.pupil_mask , calibration_amp_input = amp_input , \
    opd_internal = opd_internal,  basis = basis, Nmodes =  Nmodes, poke_amp = 0.05, poke_method = 'double_sided_poke',\
        imgs_to_mean = 1, detector=None)

IM_10 = bldr.build_IM( zwfs_ns ,  calibration_opd_input= 0 *zwfs_ns.grid.pupil_mask , calibration_amp_input = amp_input , \
    opd_internal = opd_internal,  basis = basis, Nmodes =  Nmodes, poke_amp = 0.05, poke_method = 'single_sided_poke',\
        imgs_to_mean = 1, detector=None)

# different basis 
basis = 'Hadamard'
M2C_0 = gen_basis.construct_command_basis( basis= basis, number_of_modes = Nmodes, without_piston=True).T  

IM_11 = bldr.build_IM( zwfs_ns ,  calibration_opd_input= 0 *zwfs_ns.grid.pupil_mask , calibration_amp_input = amp_input , \
    opd_internal = opd_internal,  basis = basis, Nmodes =  Nmodes, poke_amp = 0.05, poke_method = 'double_sided_poke',\
        imgs_to_mean = 1, detector=None)



################## TEST 8
# project onto TT and HO

# Build IM  and look at Eigenmodes! 
zwfs_ns = bldr.init_zwfs(grid_ns, optics_ns, dm_ns)

opd_input = 0 * zwfs_ns.grid.pupil_mask *  np.random.randn( *zwfs_ns.grid.pupil_mask.shape)

opd_internal = 10e-9 * zwfs_ns.grid.pupil_mask * np.random.randn( *zwfs_ns.grid.pupil_mask.shape)

amp_input = 1e4 * zwfs_ns.grid.pupil_mask


#basis_name_list = ['Hadamard', "Zonal", "Zonal_pinned_edges", "Zernike", "Zernike_pinned_edges", "fourier", "fourier_pinned_edges"]

# perfect field only with internal opd aberrations 
# different poke methods 
Nmodes = 100
basis = 'Zonal_pinned_edges'
poke_amp = 0.05
Smax = 30
detector = (4,4) # for binning , zwfs_ns.grid.N is #pixels across pupil diameter (64) therefore division 4 = 16 pixels (between CRed2 and Cred1 )
#M2C_0 = gen_basis.construct_command_basis( basis= basis, number_of_modes = Nmodes, without_piston=True).T  

#I0 = bldr.get_I0(  opd_input  = 0 *zwfs_ns.grid.pupil_mask ,   amp_input = amp_input,\
#    opd_internal = opd_internal,  zwfs_ns= zwfs_ns , detector=None )

#N0 = bldr.get_N0(  opd_input  = 0 *zwfs_ns.grid.pupil_mask  ,   amp_input = amp_input,\
#    opd_internal = opd_internal,  zwfs_ns= zwfs_ns , detector=None )

# we must first define our pupil regions before building 
zwfs_ns = bldr.classify_pupil_regions( opd_input,  amp_input ,  opd_internal,  zwfs_ns , detector= detector) # For now detector is just tuple of pixels to average. useful to know is zwfs_ns.grid.N is number of pixels across pupil. # from this calculate an appropiate binning for detector 

zwfs_ns = bldr.build_IM( zwfs_ns,  calibration_opd_input= 0 *zwfs_ns.grid.pupil_mask , calibration_amp_input = amp_input , \
    opd_internal = opd_internal,  basis = basis, Nmodes =  Nmodes, poke_amp = poke_amp, poke_method = 'double_sided_poke',\
        imgs_to_mean = 1, detector=detector )

# look at the eigenmodes in camera, DM and singular values
bldr.plot_eigenmodes( zwfs_ns , save_path = None )

TT_vectors = gen_basis.get_tip_tilt_vectors()

#zwfs_ns = bldr.construct_ctrl_matricies_from_IM(zwfs_ns,  method = 'Eigen_TT-HO', Smax = 50, TT_vectors = TT_vectors )
zwfs_ns = bldr.construct_ctrl_matricies_from_IM(zwfs_ns,  method = 'Eigen_TT-HO', Smax = 20, TT_vectors = TT_vectors )

#zwfs_ns = bldr.add_controllers( zwfs_ns, TT = 'PID', HO = 'leaky')
zwfs_ns = bldr.add_controllers( zwfs_ns, TT = 'PID', HO = 'leaky')

#zwfs_ns = init_CL_simulation( zwfs_ns,  opd_internal, amp_input , basis, Nmodes, poke_amp, Smax )
            
dm_disturbance = 0.1 * TT_vectors.T[0]
#zwfs_ns.dm.current_cmd =  zwfs_ns.dm.dm_flat + disturbance_cmd 

# as example how to reset telemetry
zwfs_ns.dm.current_cmd = zwfs_ns.dm.dm_flat + dm_disturbance
zwfs_ns = bldr.reset_telemetry( zwfs_ns )
zwfs_ns.ctrl.TT_ctrl.reset()
zwfs_ns.ctrl.HO_ctrl.reset()
zwfs_ns.ctrl.TT_ctrl.set_all_gains_to_zero()
zwfs_ns.ctrl.HO_ctrl.set_all_gains_to_zero()

close_after = 20
for i in range(100):
    print(f'iteration {i}')
    if i > close_after : 
        #zwfs_ns.ctrl.HO_ctrl.ki = 0.2 * np.ones( len(zwfs_ns.ctrl.HO_ctrl.ki) )
        #zwfs_ns.ctrl.HO_ctrl.kp = 1 * np.ones( len(zwfs_ns.ctrl.HO_ctrl.kp) )

        zwfs_ns.ctrl.TT_ctrl.kp = 1 * np.ones( len(zwfs_ns.ctrl.TT_ctrl.kp) )
        zwfs_ns.ctrl.TT_ctrl.ki = 0.8 * np.ones( len(zwfs_ns.ctrl.TT_ctrl.ki) )
        
    bldr.AO_iteration( opd_input, amp_input, opd_internal, zwfs_ns.reco.I0,  zwfs_ns, dm_disturbance, record_telemetry=True ,detector=detector)

# Generate some data


i = len(zwfs_ns.telem.rmse_list) - 1
plt.ioff() 
        
#for i in range(10):
im_dm_dist = util.get_DM_command_in_2D( zwfs_ns.telem.dm_disturb_list[i] )
im_phase = zwfs_ns.telem.field_phase[i]
im_int = zwfs_ns.telem.i_list[i]
im_cmd = util.get_DM_command_in_2D( np.array(zwfs_ns.telem.c_TT_list[i]) + np.array(zwfs_ns.telem.c_HO_list[i])  ) 


#line_x = np.linspace(0, i, i)
line_eHO = zwfs_ns.telem.e_HO_list[:i]
line_eTT = zwfs_ns.telem.e_TT_list[:i]
line_S = zwfs_ns.telem.strehl[:i]
line_rmse = zwfs_ns.telem.rmse_list[:i]

# Define plot data
image_list = [im_dm_dist, im_phase, im_int, im_cmd]
image_title_list = ['DM disturbance', 'input phase', 'intensity', 'reco. command']
image_colorbar_list = ['DM units', 'radians', 'adu', 'DM units']

plot_list = [ line_eHO, line_eTT, line_S, line_rmse ] 
plot_ylabel_list = ['e_HO', 'e_TT', 'Strehl', 'rmse']
plot_xlabel_list = ['iteration' for _ in plot_list]
plot_title_list = ['' for _ in plot_list]

#vlims = [(0, 1), (0, 1), (0, 1)]  # Set vmin and vmax for each image

util.create_telem_mosaic(image_list, image_title_list, image_colorbar_list, 
                plot_list, plot_title_list, plot_xlabel_list, plot_ylabel_list)













import sys
from PyQt5 import QtWidgets, QtCore
import pyqtgraph as pg


class AOControlApp(QtWidgets.QWidget):
    def __init__(self):
        super().__init__()

        # Layout
        self.layout = QtWidgets.QVBoxLayout()

        # Buttons
        self.run_button = QtWidgets.QPushButton("Run")
        self.pause_button = QtWidgets.QPushButton("Pause")
        self.zero_gains_button = QtWidgets.QPushButton("Set Gains to Zero")
        self.reset_button = QtWidgets.QPushButton("reset")
        
        # Connect buttons to functions
        self.run_button.clicked.connect(self.run_loop)
        self.pause_button.clicked.connect(self.pause_loop)
        self.zero_gains_button.clicked.connect(self.set_gains_to_zero)
        self.reset_button.clicked.connect(self.reset)

        # Text input for user
        self.input_label = QtWidgets.QLabel("User Input:")
        self.text_input = QtWidgets.QLineEdit()
        self.text_input.returnPressed.connect(self.check_input)

        # Add buttons and input to layout
        self.layout.addWidget(self.run_button)
        self.layout.addWidget(self.pause_button)
        self.layout.addWidget(self.zero_gains_button)
        self.layout.addWidget(self.reset)
        self.layout.addWidget(self.input_label)
        self.layout.addWidget(self.text_input)

        # PyQtGraph plots
        self.plot_widget = pg.GraphicsLayoutWidget()
        self.layout.addWidget(self.plot_widget)

        # Create Image and Line plots in PyQtGraph
        self.image_plots = []
        for i in range(4):
            img_view = self.plot_widget.addPlot(row=0, col=i)
            img_item = pg.ImageItem()
            img_view.addItem(img_item)
            self.image_plots.append(img_item)

        self.line_plots = []
        for i in range(4):
            line_plot = self.plot_widget.addPlot(row=1 + (i // 2), col=(i % 2) * 2, colspan=2)
            self.line_plots.append(line_plot.plot())

        # Add the layout to the widget
        self.setLayout(self.layout)

        # Timer for running the AO_iteration in a loop
        self.timer = QtCore.QTimer()
        self.timer.timeout.connect(self.run_AO_iteration)
        self.loop_running = False

    def run_loop(self):
        if not self.loop_running:
            self.loop_running = True
            self.timer.start(100)  # Adjust time in ms if needed

    def pause_loop(self):
        self.loop_running = False
        self.timer.stop()

    def set_gains_to_zero(self):
        self.pause_loop()
        zwfs_ns.ctrl.TT_ctrl.set_all_gains_to_zero()
        zwfs_ns.ctrl.HO_ctrl.set_all_gains_to_zero()
        self.run_loop()

    def reset(self):
        self.pause_loop()
        zwfs_ns = bldr.reset_telemetry( zwfs_ns )
        zwfs_ns.ctrl.TT_ctrl.reset()
        zwfs_ns.ctrl.HO_ctrl.reset()
        
    def check_input(self):
        user_input = self.text_input.text()
        self.pause_loop()
        # Placeholder: Add conditions for user input processing
        print(f"User input: {user_input}")  # Replace with actual processing logic
        
        if 'kpHO*=' in user_input:
            factor = float( user_input.split('=')[-1] )
            zwfs_ns.ctrl.HO_ctrl.kp *= factor
        if 'kpTT*=' in user_input:
            factor = float( user_input.split('=')[-1] )
            zwfs_ns.ctrl.TT_ctrl.kp *= factor
        if 'kiHO*=' in user_input:
            factor = float( user_input.split('=')[-1] )
            zwfs_ns.ctrl.HO_ctrl.ki *= factor
        if 'kiTT*=' in user_input:
            factor = float( user_input.split('=')[-1] )
            zwfs_ns.ctrl.TT_ctrl.ki *= factor

            
        self.run_loop()

    def run_AO_iteration(self):
        # Call the AO iteration function from your module
        bldr.AO_iteration( opd_input, amp_input, opd_internal, zwfs_ns.reco.I0,  zwfs_ns, dm_disturbance, record_telemetry=True ,detector=detector)


        # Retrieve telemetry data
        im_dm_dist = util.get_DM_command_in_2D(zwfs_ns.telem.dm_disturb_list[-1])
        im_phase = zwfs_ns.telem.field_phase[-1]
        im_int = zwfs_ns.telem.i_list[-1]
        im_cmd = util.get_DM_command_in_2D(np.array(zwfs_ns.telem.c_TT_list[-1]) + np.array(zwfs_ns.telem.c_HO_list[-1]))

        # Update images in the PyQtGraph interface
        self.image_plots[0].setImage(im_dm_dist)
        self.image_plots[1].setImage(im_phase)
        self.image_plots[2].setImage(im_int)
        self.image_plots[3].setImage(im_cmd)

        # Update line plots
        # Check if line data exists
        if len(zwfs_ns.telem.e_HO_list) > 0:
            self.line_plots[0].setData(zwfs_ns.telem.e_HO_list)
            self.line_plots[0].getViewBox().autoRange()  # Force autoscaling of the plot

        if len(zwfs_ns.telem.e_TT_list) > 0:
            self.line_plots[1].setData(zwfs_ns.telem.e_TT_list)
            self.line_plots[1].getViewBox().autoRange()

        if len(zwfs_ns.telem.strehl) > 0:
            self.line_plots[2].setData(zwfs_ns.telem.strehl)
            self.line_plots[2].getViewBox().autoRange()

        if len(zwfs_ns.telem.rmse_list) > 0:
            self.line_plots[3].setData(zwfs_ns.telem.rmse_list)
            self.line_plots[3].getViewBox().autoRange()


if __name__ == "__main__":
    
    zwfs_ns = bldr.reset_telemetry( zwfs_ns )
    zwfs_ns.ctrl.TT_ctrl.reset()
    zwfs_ns.ctrl.HO_ctrl.reset()
    zwfs_ns.ctrl.TT_ctrl.set_all_gains_to_zero()
    zwfs_ns.ctrl.HO_ctrl.set_all_gains_to_zero()

    zwfs_ns.ctrl.HO_ctrl.ki = 0.2 * np.ones( len(zwfs_ns.ctrl.HO_ctrl.ki) )
    zwfs_ns.ctrl.HO_ctrl.kp = 1 * np.ones( len(zwfs_ns.ctrl.HO_ctrl.kp) )

    zwfs_ns.ctrl.TT_ctrl.kp = 1 * np.ones( len(zwfs_ns.ctrl.TT_ctrl.kp) )
    zwfs_ns.ctrl.TT_ctrl.ki = 0.8 * np.ones( len(zwfs_ns.ctrl.TT_ctrl.ki) )
    


    app = QtWidgets.QApplication(sys.argv)
    window = AOControlApp()
    window.setWindowTitle("AO Control GUI")
    window.show()
    sys.exit(app.exec_())












"""
import pyqtgraph as pg
from pyqtgraph.Qt import QtCore, QtWidgets
import numpy as np
import sys

# Data placeholders for demonstration
dm_disturb_list = [np.random.randn(12, 12) for _ in range(100)]
i_list = [np.random.rand(64, 64) for _ in range(100)]
e_TT_list = [np.random.randn(10) for _ in range(100)]
e_HO_list = [np.random.randn(20) for _ in range(100)]
rmse_list = np.random.rand(100)

# Create a Qt Application
app = QtWidgets.QApplication([])

# Create the main window and layout
main_win = QtWidgets.QWidget()
main_layout = QtWidgets.QVBoxLayout()
main_win.setLayout(main_layout)

# Create the pyqtgraph GraphicsLayoutWidget for plots
plot_win = pg.GraphicsLayoutWidget(show=True, title="Real-Time Plotting with PyQtGraph")
plot_win.resize(1000, 800)
plot_win.setWindowTitle('Real-Time ZWFS Control Loops')

main_layout.addWidget(plot_win)

# Create the image plot for DM disturbance
dm_plot = plot_win.addPlot(title="DM Disturbance")
dm_img = pg.ImageItem()
dm_plot.addItem(dm_img)

# Create the image plot for i_list (camera image)
i_list_plot = plot_win.addPlot(title="Camera Image")
i_img = pg.ImageItem()
i_list_plot.addItem(i_img)
plot_win.nextRow()

# Create the line plot for e_TT signal
e_TT_plot = plot_win.addPlot(title="Tip/Tilt Error Signal (e_TT)")
e_TT_curve = e_TT_plot.plot()

# Create the line plot for e_HO signal
e_HO_plot = plot_win.addPlot(title="Higher Order Error Signal (e_HO)")
e_HO_curve = e_HO_plot.plot()
plot_win.nextRow()

# Create the line plot for RMSE
rmse_plot = plot_win.addPlot(title="RMSE")
rmse_curve = rmse_plot.plot()

# Now create a horizontal layout for buttons
button_layout = QtWidgets.QHBoxLayout()

# Run Button
run_button = QtWidgets.QPushButton('Run')
button_layout.addWidget(run_button)

# Pause Button
pause_button = QtWidgets.QPushButton('Pause')
button_layout.addWidget(pause_button)

# Reset Button
reset_button = QtWidgets.QPushButton('Reset')
button_layout.addWidget(reset_button)

# Add the button panel to the main layout
main_layout.addLayout(button_layout)

# Initialize some settings
idx = 0
paused = False  # To handle pause

def update_plots(i):
    # Update DM disturbance image
    dm_img.setImage(dm_disturb_list[i])

    # Update i_list (camera image)
    i_img.setImage(i_list[i])

    # Update e_TT signal (1D)
    e_TT_curve.setData(e_TT_list[i])

    # Update e_HO signal (1D)
    e_HO_curve.setData(e_HO_list[i])

    # Update RMSE curve
    rmse_curve.setData(rmse_list[:i+1])

# Timer to update the plots in real-time
def update():
    global idx, paused
    if paused:
        return  # Do nothing if paused
    if idx >= len(i_list):  # Stop when finished
        return
    update_plots(idx)
    idx += 1

# Run button event handler
def run():
    global paused
    paused = False  # Resume the update
    timer.start(100)

# Pause button event handler
def pause():
    global paused
    paused = True  # Stop the update

# Reset button event handler
def reset():
    global idx, paused
    idx = 0  # Reset index to start over
    paused = True  # Stop the update
    update_plots(idx)  # Re-initialize the first frame

# Connect buttons to their functions
run_button.clicked.connect(run)
pause_button.clicked.connect(pause)
reset_button.clicked.connect(reset)

# Start a QTimer to call the update function periodically
timer = QtCore.QTimer()
timer.timeout.connect(update)
timer.start(100)  # Update every 100ms (10 frames per second)

# Show the main window
main_win.show()

# Start the Qt event loop
if __name__ == '__main__':
    if (sys.flags.interactive != 1) or not hasattr(QtCore, 'PYQT_VERSION'):
        QtWidgets.QApplication.instance().exec_()

"""






