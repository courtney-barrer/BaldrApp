import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider

def generate_synthetic_image(phasemask_center, pupil_mask, image_size=(256, 256), pupil_diameter=200):
    """
    Generate a synthetic image for the Zernike wavefront sensor.

    Parameters:
        phasemask_center (tuple): (x, y) position of the phasemask center.
        pupil_mask (ndarray): Boolean array representing the active pupil.
        image_size (tuple): Dimensions of the image (height, width).
        pupil_diameter (int): Diameter of the pupil.

    Returns:
        ndarray: Synthetic image with the phasemask misaligned.
    """
    y, x = np.indices(image_size)
    cx, cy = phasemask_center

    # Gaussian with 1/e radius equal to pupil radius
    sigma = pupil_diameter / 1.5 / np.sqrt(2)
    gaussian = np.exp(-((x - cx)**2 + (y - cy)**2) / (2 * sigma**2))
    synthetic_image = gaussian * pupil_mask + 0.1*np.random.randn(*gaussian.shape)

    return synthetic_image

def split_into_quadrants(image, pupil_mask):
    """
    Split the image into four quadrants using the active pupil mask.

    Parameters:
        image (ndarray): Input image.
        pupil_mask (ndarray): Boolean array representing the active pupil.

    Returns:
        dict: Dictionary of quadrants (top-left, top-right, bottom-left, bottom-right).
    """
    y, x = np.indices(image.shape)
    cx, cy = np.mean(np.where(pupil_mask), axis=1).astype(int)

    # Create boolean masks for each quadrant
    top_left_mask = (y < cy) & (x < cx) & pupil_mask
    top_right_mask = (y < cy) & (x >= cx) & pupil_mask
    bottom_left_mask = (y >= cy) & (x < cx) & pupil_mask
    bottom_right_mask = (y >= cy) & (x >= cx) & pupil_mask

    quadrants = {
        "top_left": image[top_left_mask],
        "top_right": image[top_right_mask],
        "bottom_left": image[bottom_left_mask],
        "bottom_right": image[bottom_right_mask],
    }

    return quadrants

def weighted_photometric_difference(quadrants):
    """
    Calculate the weighted photometric difference between quadrants.

    Parameters:
        quadrants (dict): Dictionary of quadrants.

    Returns:
        tuple: (x_error, y_error) error vectors.
    """
    top = np.sum(quadrants["top_left"]) + np.sum(quadrants["top_right"])
    bottom = np.sum(quadrants["bottom_left"]) + np.sum(quadrants["bottom_right"])

    left = np.sum(quadrants["top_left"]) + np.sum(quadrants["bottom_left"])
    right = np.sum(quadrants["top_right"]) + np.sum(quadrants["bottom_right"])

    y_error = top - bottom
    x_error = left - right

    return x_error, y_error

def closed_loop_simulation(pupil_mask, image_size=(256, 256), pupil_diameter=200, gain = 100, max_iterations=50, tolerance=1e-3):
    """
    Perform a closed-loop simulation to align the phasemask.

    Parameters:
        pupil_mask (ndarray): Boolean array representing the active pupil.
        image_size (tuple): Dimensions of the image (height, width).
        pupil_diameter (int): Diameter of the pupil.
        max_iterations (int): Maximum number of iterations.
        tolerance (float): Error tolerance for convergence.

    Returns:
        list: History of phasemask positions and synthetic images.
    """
    y, x = np.indices(image_size)
    cx, cy = image_size[0] // 2, image_size[1] // 2
    phasemask_center = [cx + 20, cy - 20]  # Initial misaligned position
    history = [tuple(phasemask_center)]
    images = []

    for iteration in range(max_iterations):
        synthetic_image = generate_synthetic_image(phasemask_center, pupil_mask, image_size, pupil_diameter)
        images.append(synthetic_image)
        quadrants = split_into_quadrants(synthetic_image, pupil_mask)
        x_error, y_error = weighted_photometric_difference(quadrants)

        # Update phasemask center
        phasemask_center[0] += gain * x_error / np.sum(pupil_mask)
        phasemask_center[1] += gain * y_error / np.sum(pupil_mask)
        history.append(tuple(phasemask_center))

        # Check for convergence
        if np.sqrt(x_error**2 + y_error**2) < tolerance:
            print(f"Converged in {iteration + 1} iterations.")
            break

    return history, images

# Example usage
image_size = (256, 256)
pupil_diameter = 200

# Create a circular pupil mask
y, x = np.indices(image_size)
cx, cy = image_size[0] // 2, image_size[1] // 2
distance = np.sqrt((x - cx)**2 + (y - cy)**2)
pupil_mask = distance <= (pupil_diameter / 2)

# Run closed-loop simulation
history, images = closed_loop_simulation(pupil_mask, image_size, pupil_diameter, max_iterations=150, tolerance=1e-6)

# Interactive plot with slider
positions = np.array(history)

fig, ax = plt.subplots(1, 2, figsize=(12, 6))
plt.subplots_adjust(bottom=0.2)

# Initialize plots
image_plot = ax[0].imshow(images[0], cmap='hot', extent=(0, image_size[1], 0, image_size[0]))
ax[0].set_title("Synthetic Image")
position_plot, = ax[1].plot(positions[:, 0], positions[:, 1], marker='o', linestyle='-', color='blue', alpha=0.5)
current_position, = ax[1].plot(positions[0, 0], positions[0, 1], marker='o', color='red')
ax[1].set_xlim(positions[:, 0].min() - 5, positions[:, 0].max() + 5)
ax[1].set_ylim(positions[:, 1].min() - 5, positions[:, 1].max() + 5)
ax[1].set_title("Phasemask Center History")
ax[1].set_xlabel("x position")
ax[1].set_ylabel("y position")
ax[1].grid()

# Slider setup
ax_slider = plt.axes([0.2, 0.05, 0.65, 0.03])
slider = Slider(ax_slider, "Iteration", 0, len(images) - 1, valinit=0, valstep=1)

# Update function for slider
def update(val):
    idx = int(slider.val)
    image_plot.set_data(images[idx])
    current_position.set_data([positions[idx, 0]], [positions[idx, 1]])
    fig.canvas.draw_idle()

slider.on_changed(update)
plt.show()



def plot_quadrants(pupil_mask):
    """
    Plot the clear pupil with quadrants overlaid for visualization.
    """
    y, x = np.indices(pupil_mask.shape)
    cx, cy = np.mean(np.where(pupil_mask), axis=1).astype(int)

    # Generate the quadrants as 2D masks
    top_left_mask = ((y < cy) & (x < cx) & pupil_mask).astype(float)
    top_right_mask = ((y < cy) & (x >= cx) & pupil_mask).astype(float)
    bottom_left_mask = ((y >= cy) & (x < cx) & pupil_mask).astype(float)
    bottom_right_mask = ((y >= cy) & (x >= cx) & pupil_mask).astype(float)

    # Plot the clear pupil and overlay quadrants
    fig, ax = plt.subplots()
    ax.imshow(pupil_mask.astype(float), cmap='gray')
    ax.contour(top_left_mask, levels=[0.5], colors='red', linewidths=1)
    ax.contour(top_right_mask, levels=[0.5], colors='green', linewidths=1)
    ax.contour(bottom_left_mask, levels=[0.5], colors='blue', linewidths=1)
    ax.contour(bottom_right_mask, levels=[0.5], colors='yellow', linewidths=1)
    ax.set_title("Clear Pupil with Quadrants Overlaid")
    plt.show()

# Call the function to plot
plot_quadrants(pupil_mask)








# import zmq
# import toml  # Make sure to install via `pip install toml` if needed
# import argparse
# import os
# import json
# import time

# from xaosim.shmlib import shm
# import asgard_alignment.controllino as co # for turning on / off source 
# import common.DM_registration as DM_registration
# try:
#     from asgard_alignment import controllino as co
#     myco = co.Controllino('172.16.8.200')
#     controllino_available = True
#     print('controllino connected')
    
# except:
#     print('WARNING Controllino cannot connect. WILL NOT MOVE SOURCE OUT FOR DARK')
#     controllino_available = False 
# """
# idea it to be able to align phasemask position 
# in a mode independent way with significant focus offsets
# using image symmetry across registered pupil as objective 
# """


# def send_and_get_response(message):
#     # st.write(f"Sending message to server: {message}")
#     state_dict["message_history"].append(
#         f":blue[Sending message to server: ] {message}\n"
#     )
#     state_dict["socket"].send_string(message)
#     response = state_dict["socket"].recv_string()
#     if "NACK" in response or "not connected" in response:
#         colour = "red"
#     else:
#         colour = "green"
#     # st.markdown(f":{colour}[Received response from server: ] {response}")
#     state_dict["message_history"].append(
#         f":{colour}[Received response from server: ] {response}\n"
#     )

#     return response.strip()



# def plot_telemetry(telemetry, savepath=None):
#     """
#     Plots the phasemask centering telemetry for each beam.
    
#     Parameters:
#         telemetry (dict): A dictionary where keys are beam IDs and values are dictionaries
#                           with keys:
#                               "phasmask_Xpos" - list of X positions,
#                               "phasmask_Ypos" - list of Y positions,
#                               "phasmask_Xerr" - list of X errors,
#                               "phasmask_Yerr" - list of Y errors.
#     """
#     for beam_id, data in telemetry.items():
#         # Determine the number of iterations
#         num_iterations = len(data["phasmask_Xpos"])
#         iterations = np.arange(1, num_iterations + 1)
        
#         # Create a figure with two subplots: one for positions and one for errors
#         fig, axs = plt.subplots(1, 2, figsize=(12, 5))
#         fig.suptitle(f"Telemetry for Beam {beam_id}", fontsize=14)
        
#         # Plot phasemask positions
#         axs[0].plot(iterations, data["phasmask_Xpos"], marker='o', label="X Position")
#         axs[0].plot(iterations, data["phasmask_Ypos"], marker='s', label="Y Position")
#         axs[0].set_xlabel("Iteration")
#         axs[0].set_ylabel("Position (um)")
#         axs[0].set_title("Phasemask Positions")
#         axs[0].legend()
#         axs[0].grid(True)
        
#         # Plot phasemask errors
#         axs[1].plot(iterations, data["phasmask_Xerr"], marker='o', label="X Error")
#         axs[1].plot(iterations, data["phasmask_Yerr"], marker='s', label="Y Error")
#         axs[1].set_xlabel("Iteration")
#         axs[1].set_ylabel("Error (um)")
#         axs[1].set_title("Phasemask Errors")
#         axs[1].legend()
#         axs[1].grid(True)
        
#         plt.tight_layout(rect=[0, 0, 1, 0.95])
#         if savepath is not None:
#             plt.savefig(savepath)
#         plt.show()



# def image_slideshow(telemetry, beam_id):

#     # Interactive plot with slider
#     positions = [(x,y) for x,y in zip(telemetry[beam_id]["phasmask_Xpos"],telemetry[beam_id]["phasmask_Ypos"])]
#     images = telemetry[beam_id]["img"]
#     fig, ax = plt.subplots(1, 2, figsize=(12, 6))
#     plt.subplots_adjust(bottom=0.2)

#     # Initialize plots
#     image_plot = ax[0].imshow(images[0], cmap='hot')
#     ax[0].set_title("Image")
#     position_plot, = ax[1].plot(positions[:, 0], positions[:, 1], marker='o', linestyle='-', color='blue', alpha=0.5)
#     current_position, = ax[1].plot(positions[0, 0], positions[0, 1], marker='o', color='red')
#     ax[1].set_xlim(positions[:, 0].min() - 5, positions[:, 0].max() + 5)
#     ax[1].set_ylim(positions[:, 1].min() - 5, positions[:, 1].max() + 5)
#     ax[1].set_title("Phasemask Center History")
#     ax[1].set_xlabel("x position")
#     ax[1].set_ylabel("y position")
#     ax[1].grid()

#     # Slider setup
#     ax_slider = plt.axes([0.2, 0.05, 0.65, 0.03])
#     slider = Slider(ax_slider, "Iteration", 0, len(images) - 1, valinit=0, valstep=1)

#     # Update function for slider
#     def update(val):
#         idx = int(slider.val)
#         image_plot.set_data(images[idx])
#         current_position.set_data([positions[idx, 0]], [positions[idx, 1]])
#         fig.canvas.draw_idle()

#     slider.on_changed(update)
#     plt.show()


# parser = argparse.ArgumentParser(description="Baldr phase mask fine x-y adjustment")


# # Camera shared memory path
# parser.add_argument(
#     "--camera_shm",
#     type=str,
#     default="/dev/shm/cred1.im.shm",
#     help="Camera shared memory path. Default: /dev/shm/cred1.im.shm"
# )

# # TOML file path; default is relative to the current file's directory.
# default_toml = os.path.join("config_files", "baldr_config.toml") #os.path.dirname(os.path.abspath(__file__)), "..", "config_files", "baldr_config.toml")
# parser.add_argument(
#     "--toml_file",
#     type=str,
#     default=default_toml,
#     help="TOML file to write/edit. Default: ../config_files/baldr_config.toml (relative to script)"
# )

# # Beam ids: provided as a comma-separated string and converted to a list of ints.
# parser.add_argument(
#     "--beam_id",
#     type=lambda s: [int(item) for item in s.split(",")],
#     default=[1, 2, 3, 4],
#     help="Comma-separated list of beam IDs. Default: 1,2,3,4"
# )

# parser.add_argument(
#     "--max_iterations",
#     type=int,
#     default=10,
#     help="maximum number of iterations allowed in centering. Default = 10"
# )

# parser.add_argument(
#     "--gain",
#     type=int,
#     default=0.1,
#     help="gain to be applied for centering beam. Default = 0.1 "
# )

# parser.add_argument(
#     "--tol",
#     type=int,
#     default=0.1,
#     help="tolerence for convergence of centering algorithm. Default = 0.1 "
# )

# # Plot: default is True, with an option to disable.
# parser.add_argument(
#     "--plot", 
#     dest="plot",
#     action="store_true",
#     default=True,
#     help="Enable plotting (default: True)"
# )


# args = parser.parse_args()

# # set up commands to move motors phasemask
# context = zmq.Context()
# context.socket(zmq.REQ)
# socket = context.socket(zmq.REQ)
# socket.setsockopt(zmq.RCVTIMEO, args.timeout)
# server_address = f"tcp://{args.host}:{args.port}"
# socket.connect(server_address)
# state_dict = {"message_history": [], "socket": socket}

# # phasemask specific commands
# # message = f"fpm_movetomask phasemask{args.beam} {args.phasemask_name}"
# # res = send_and_get_response(message)
# # print(res)

# for beam_id in args.beam_id:
#     message = f"read BMX{beam_id}"
#     Xpos = float( send_and_get_response(message) )

#     message = f"read BMY{beam_id}"
#     Ypos = float( send_and_get_response(message) )

#     print('starting from current positiom X={}, Y={}um on beam {beam_id}')
#     phasemask_center = [Xpos, Ypos]

# #example to move x-y of each beam's phasemask 
# for beam_id in args.beam_id:


# # set up commands to move DM 
# assert hasattr(args.beam_id , "__len__")
# assert len(args.beam_id) <= 4
# assert max(args.beam_id) <= 4
# assert min(args.beam_id) >= 1 

# dm_shm_dict = {}
# for beam_id in args.beam_id:
#     dm_shm_dict[beam_id] = dmclass( beam_id=beam_id )
#     # zero all channels
#     dm_shm_dict[beam_id].zero_all()
#     # activate flat 
#     dm_shm_dict[beam_id].activate_flat()

# # set up camera 
# c = shm(args.global_camera_shm)

# # set up subpupils and pixel mask
# with open(args.toml_file ) as file:
#     pupildata = toml.load(file)
#     # Extract the "baldr_pupils" section
#     baldr_pupils = pupildata.get("baldr_pupils", {})

#     # the registered pupil mask for each beam (in the local frame)
#     pupil_masks={}
#     for beam_id in args.beam_id:
#         pupil_masks[beam_id] = pupildata.get("beam{beam_id}.pupil_mask.mask")



# # dark and badpixel mask
# if controllino_available:

#     myco.turn_off("SBB")
#     time.sleep(2)
    
#     dark_raw = c.get_data()

#     myco.turn_on("SBB")
#     time.sleep(2)

#     bad_pixel_mask = get_bad_pixel_indicies( dark_raw, std_threshold = 20, mean_threshold=6)
# else:
#     dark_raw = c.get_data()

#     bad_pixel_mask = get_bad_pixel_indicies( dark_raw, std_threshold = 20, mean_threshold=6)



# # get initial image
# img = c.get_data() #  full image 
# initial_images = {}
# for beam_id in args.beam_id:
#     r1,r2,c1,c2 = baldr_pupils[f"{beam_id}"]
#     cropped_img = interpolate_bad_pixels(img[r1:r2, c1:c2], bad_pixel_mask[r1:r2, c1:c2])
#     initial_images[beam_id] = cropped_img

# # begin centering algorithm, tracking telemetry
# telemetry={b:{"phasmask_Xpos":[],"phasmask_Ypos":[],"phasmask_Xerr":[], "phasmask_Yerr":[]} for b in args.beam_id }

# complete_flag={b:False for b in args.beam_id}

# for iteration in range(args.max_iterations):

#     # get image 
#     img = c.get_data() # full image 

#     for beam_id in args.beam_id:
#         if not complete_flag[beam_id]:
#             r1,r2,c1,c2 = baldr_pupils[f"{beam_id}"]
#             cropped_img = interpolate_bad_pixels(img[r1:r2, c1:c2], bad_pixel_mask[r1:r2, c1:c2])
#             # normalize by the mean within defined pupil mask
#             cropped_img *= 1/np.mean(cropped_img[pupil_masks[beam_id]])

#             quadrants = split_into_quadrants(cropped_img, pupil_mask)
#             x_error, y_error = weighted_photometric_difference(quadrants)

#             # Update phasemask center
#             phasemask_center[0] += args.gain * x_error / np.sum(pupil_mask)
#             phasemask_center[1] += args.gain * y_error / np.sum(pupil_mask)

#             # Move 
#             message = f"moveabs BMX{beam_id} {phasemask_center[0]}"
#             ok = float( send_and_get_response(message) )
#             print(ok)
#             message = f"read BMY{beam_id} {phasemask_center[1]}"
#             ok = float( send_and_get_response(message) )
#             print(ok)
            
#             # Check for convergence
#             if np.sqrt(x_error**2 + y_error**2) < args.tolerance:
#                 print(f"Beam {beam_id} converged in {iteration + 1} iterations.")
#                 complete_flag[beam_id] = True

#             telemetry[beam_id]["img"] =  cropped_img
#             telemetry[beam_id]["phasmask_Xpos"] = phasemask_center[0]
#             telemetry[beam_id]["phasmask_Ypos"] = phasemask_center[1]
#             telemetry[beam_id]["phasmask_Xerr"] = x_error
#             telemetry[beam_id]["phasmask_Yerr"] = y_error


# # get final image after convergence 
# img = c.get_data() #  full image 
# final_images = {}
# for beam_id in args.beam_id:
#     r1,r2,c1,c2 = baldr_pupils[f"{beam_id}"]
#     cropped_img = interpolate_bad_pixels(img[r1:r2, c1:c2], bad_pixel_mask[r1:r2, c1:c2])
#     final_images[beam_id] = cropped_img

# # some diagnostic plots  
# plot_telemetry(telemetry, savepath=None)

# # slideshow of images for a beam
# #image_slideshow(telemetry, beam_id)