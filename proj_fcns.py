import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import os

def load_images(folder, label, img_size=(128, 128)):
    """
    Loads images from a folder and assigns them a label.

    Parameters:
        folder (str): Directory containing images.
        label (0 or 1): Label assigned to each image.
        img_size (tuple): dimentions to resize images (width, height). Default is (128, 128).

    Returns:
        tuple: Lists of preprocessed images and their labels.
    """
    images = []
    labels = []
    for filename in os.listdir(folder):
        if filename.endswith(".jpg") or filename.endswith(".jpeg") or filename.endswith(".png"):
            img_path = os.path.join(folder, filename)
            img = Image.open(img_path).convert('RGB')  # Convert to RGB
            
            # Crop the image to a square
            width, height = img.size
            if width != height:
                if width > height:
                    left = (width - height) // 2
                    right = left + height
                    top = 0
                    bottom = height
                else:
                    top = (height - width) // 2
                    bottom = top + width
                    left = 0
                    right = width
                img = img.crop((left, top, right, bottom))
            
            # Resize the image to the desired size
            img = img.resize(img_size)
            
            img_array = np.array(img)
            images.append(img_array)
            labels.append(label)
    return images, labels

def sigmoid(z):
    """
    Computes the sigmoid function, it maps any real number into the range (0, 1).

    Parameters:
        z (array or scalar): Input values, clipped to [-500, 500].

    Returns:
        array-like or scalar: Sigmoid of the input values.
    """
    z = np.clip(z, -500, 500)  # Clip values to avoid overflow
    return 1 / (1 + np.exp(-z))

def initialize_parameters(dim):
    """
    Initializes parameters for a model.

    Parameters:
        dim (int): Size of the weight vector.

    Returns:
        tuple: Initialized weight vector (w) and bias (b).
    """
    np.random.seed(1)
    w = np.random.randn(dim, 1) * 0.01  # Small random values - was behaving oddly with zeros
    b = 0
    return w, b

def plot_costs(data):
    """
    Plots the cost reduction over iterations with specified styles.

    Parameters:
    data (list or array): The cost data to be plotted.
    """
    # Create a new figure and axis with the specified background color
    fig, ax = plt.subplots()

    # Set the background color for the whole plot, including labels
    fig.patch.set_facecolor('#d6d0c1')  # Light green background for the whole plot
    ax.set_facecolor('#d6d0c1')  # Light green background for the plot area

    # Plot the data with the custom line color
    ax.plot(data, color='#2e541a')

    # Set labels and title with typewriter style font
    plt.ylabel('Cost', fontname='Courier New', color='#4b2e00')  # Dark brown font color
    plt.xlabel('Iterations (hundreds)', fontname='Courier New', color='#4b2e00')  # Dark brown font color
    plt.title('Cost reduction over iterations', fontname='Courier New', color='#4b2e00')  # Dark brown font color

    # Customize grid lines
    ax.grid(color='#4b2e00', linestyle='-', linewidth=0.7)  # Light brown grid lines

    # Customize tick labels to be in dark brown font color
    ax.tick_params(axis='both', colors='#4b2e00')  # Dark brown tick labels

    # Customize the border colors to be dark brown
    ax.spines['top'].set_color('#4b2e00')
    ax.spines['bottom'].set_color('#4b2e00')
    ax.spines['left'].set_color('#4b2e00')
    ax.spines['right'].set_color('#4b2e00')

    # Set the axis line color to be dark brown
    ax.xaxis.label.set_color('#4b2e00')
    ax.yaxis.label.set_color('#4b2e00')

    # Set the title color to be dark brown
    ax.title.set_color('#4b2e00')

    # Show the plot
    plt.show()