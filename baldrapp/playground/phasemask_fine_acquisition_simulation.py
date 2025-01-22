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
    sigma = pupil_diameter / 2 / np.sqrt(2)
    gaussian = np.exp(-((x - cx)**2 + (y - cy)**2) / (2 * sigma**2))
    synthetic_image = gaussian * pupil_mask

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
