import cv2
import numpy as np
import matplotlib.pyplot as plt

def plot_histogram(image: np.ndarray):
    """
    Draw histograms (Pixel value and Frequency) of an image.
    :param image: array data from image
    :return: None
    """
    # Check if image is loaded successfully
    if image is None:
        raise FileNotFoundError("Image could not be loaded")

    # Create a figure
    fig, ax = plt.subplots(figsize=(7, 5))

    # Check if image is grayscale or RGB
    if image.ndim == 2:  # Grayscale
        hist = cv2.calcHist([image], [0], None, [256], [0, 256]).flatten()
        ax.bar(range(256), hist, color='black')
        ax.set_title('Grayscale Histogram')
        ax.set_xlim([0, 256])
    elif image.ndim == 3:  # RGB
        colors = ('b', 'g', 'r')
        for i, color in enumerate(colors):
            hist = cv2.calcHist([image], [i], None, [256], [0, 256]).flatten()
            ax.bar(range(256), hist, color=color, alpha=0.5, label=color)
        ax.set_title('RGB Histogram')
        ax.set_xlim([0, 256])
        ax.legend()

    ax.set_xlabel('Pixel value')
    ax.set_ylabel('Frequency')
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    image_path = 'E:/Study/pythonProject/Image/Original/1.png'
    image = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
    plot_histogram(image)
