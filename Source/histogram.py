import cv2
import numpy as np
import matplotlib.pyplot as plt


def plot_histograms(original_img: np.ndarray, watermarked_img: np.ndarray):
    """
    Draw 2 histogram (Pixel value and Frequency) of original and watermarked image
    :param original_img: array data from original image
    :param watermarked_img: array data from watermarked image
    :return: None
    """
    # Check if images are loaded successfully
    if original_img is None:
        raise FileNotFoundError(f"Original image at path {original_image_path} could not be loaded")
    if watermarked_img is None:
        raise FileNotFoundError(f"Watermarked image at path {watermarked_image_path} could not be loaded")

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    if original_img.ndim == 2:  # Grayscale
        hist = cv2.calcHist([original_img], [0], None, [256], [0, 256])
        axes[0].plot(hist, color='black')
        axes[0].set_title('Grayscale Histogram: Original Image')
        axes[0].set_xlim([0, 256])
    elif original_img.ndim == 3:  # RGB
        colors = ('b', 'g', 'r')
        for i, color in enumerate(colors):
            hist = cv2.calcHist([original_img], [i], None, [256], [0, 256])
            axes[0].plot(hist, color=color)
        axes[0].set_title('RGB Histogram: Original Image')
        axes[0].set_xlim([0, 256])

    if watermarked_img.ndim == 2:
        hist = cv2.calcHist([watermarked_img], [0], None, [256], [0, 256])
        axes[1].plot(hist, color='black')
        axes[1].set_title('Grayscale Histogram: Watermarked Image')
        axes[1].set_xlim([0, 256])
    elif watermarked_img.ndim == 3:
        colors = ('b', 'g', 'r')
        for i, color in enumerate(colors):
            hist = cv2.calcHist([watermarked_img], [i], None, [256], [0, 256])
            axes[1].plot(hist, color=color)
        axes[1].set_title('RGB Histogram: Watermarked Image')
        axes[1].set_xlim([0, 256])

    for ax in axes:
        ax.set_xlabel('Pixel value')
        ax.set_ylabel('Frequency')

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    original_image_path = 'E:/Study/pythonProject/Image/Original/1.png'
    watermarked_image_path = 'E:/Study/pythonProject/Image/test.png'
    original_img = cv2.imread(original_image_path, cv2.IMREAD_UNCHANGED)
    watermarked_img = cv2.imread(watermarked_image_path, cv2.IMREAD_UNCHANGED)
    plot_histograms(original_img, watermarked_img)
