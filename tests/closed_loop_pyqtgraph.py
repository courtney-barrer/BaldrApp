
import numpy as np
import matplotlib.pyplot as plt
from types import SimpleNamespace
import importlib 
from matplotlib.gridspec import GridSpec
from mpl_toolkits.axes_grid1 import make_axes_locatable

import Baldr_closeloop as bldr
import DM_basis as gen_basis
import utilities as util

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
        #self.reset_button = QtWidgets.QPushButton("reset")
        
        # Visual "LED" for running state
        self.status_led = QtWidgets.QLabel()
        self.status_led.setFixedSize(20, 20)
        self.update_led(False)  # Initialize as "not running" (False)

        # Connect buttons to functions
        self.run_button.clicked.connect(self.run_loop)
        self.pause_button.clicked.connect(self.pause_loop)
        self.zero_gains_button.clicked.connect(self.set_gains_to_zero)
        #self.reset_button.clicked.connect(self.reset)

        # Text input for user
        self.input_label = QtWidgets.QLabel("User Input:")
        self.text_input = QtWidgets.QLineEdit()
        self.text_input.returnPressed.connect(self.check_input)

        # Add buttons and input to layout
        self.layout.addWidget(self.run_button)
        self.layout.addWidget(self.pause_button)
        self.layout.addWidget(self.zero_gains_button)
        #self.layout.addWidget(self.reset_button)
        self.layout.addWidget(self.input_label)
        self.layout.addWidget(self.text_input)
        self.layout.addWidget(self.status_led)  # Add LED to the layout
        
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

    #def reset(self):
    #    self.pause_loop()
    #    zwfs_ns = bldr.reset_telemetry( zwfs_ns )
    #    zwfs_ns.ctrl.TT_ctrl.reset()
    #    zwfs_ns.ctrl.HO_ctrl.reset()
        
    def update_led(self, running):
        """Update the LED color depending on whether the system is running."""
        if running:
            self.status_led.setStyleSheet("background-color: green; border-radius: 10px;")
        else:
            self.status_led.setStyleSheet("background-color: red; border-radius: 10px;")


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

        
        if 'kpHO[' in user_input:
            index = int( user_input.split('[')[1].split(']')[0] )
            value = float( user_input.split('=')[-1] )
            zwfs_ns.ctrl.HO_ctrl.kp[index] = value
        if 'kpTT[' in user_input:
            index = int( user_input.split('[')[1].split(']')[0] )
            value = float( user_input.split('=')[-1] )
            zwfs_ns.ctrl.TT_ctrl.kp[index] = value
        if 'kiHO[' in user_input:
            index = int( user_input.split('[')[1].split(']')[0] )
            value = float( user_input.split('=')[-1] )
            zwfs_ns.ctrl.HO_ctrl.ki[index] = value
        if 'kiTT[' in user_input:
            index = int( user_input.split('[')[1].split(']')[0] )
            value = float( user_input.split('=')[-1] )
            zwfs_ns.ctrl.TT_ctrl.ki[index] = value

                    
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


# Build IM  and look at Eigenmodes! 
zwfs_ns = bldr.init_zwfs(grid_ns, optics_ns, dm_ns)

opd_input = 0 * zwfs_ns.grid.pupil_mask *  np.random.randn( *zwfs_ns.grid.pupil_mask.shape)

opd_internal = 10e-9 * zwfs_ns.grid.pupil_mask * np.random.randn( *zwfs_ns.grid.pupil_mask.shape)

amp_input = 1e4 * zwfs_ns.grid.pupil_mask


#basis_name_list = ['Hadamard', "Zonal", "Zonal_pinned_edges", "Zernike", "Zernike_pinned_edges", "fourier", "fourier_pinned_edges"]

# perfect field only with internal opd aberrations 
# different poke methods 
Nmodes = 100
# ['Hadamard', "Zonal", "Zonal_pinned_edges", "Zernike", "Zernike_pinned_edges", "fourier", "fourier_pinned_edges"]
basis = 'Hadamard' #'Zonal_pinned_edges'
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







#####################
## REAL TIME SIMULATION APP 
####################

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
