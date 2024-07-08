import cv2
import numpy as np
from enum import Enum


class Method(Enum):
    DIRECT = 1
    BITWISE_ADD = 2
    NEGATED_BITWISE_ADD = 3


def method_embed(container_pixel: np.ndarray, watermark_pixel: np.ndarray, method: Method) -> np.uint8:
    """
    Change last bit in last bit plane container
    :param container_pixel: data container
    :param watermark_pixel: data watermark image
    :param method: 3 methods (DIRECT, BITWISE_ADD, NEGATED_BITWISE_ADD) for 3 methods
    :return: plane after changing
    """

    # Extract the last bit of the container pixel
    container_last_bit = container_pixel & 1
    # Extract the last bit of the watermark pixel
    watermark_last_bit = watermark_pixel & 1

    match method:
        case Method.DIRECT:
            container_pixel = ((container_pixel & 0xFE) | watermark_last_bit)
        case Method.BITWISE_ADD:
            new_bit = container_last_bit ^ watermark_last_bit
            container_pixel = np.uint8((container_pixel & 0xFE) | new_bit)
        case Method.NEGATED_BITWISE_ADD:
            new_bit = ~(container_last_bit ^ watermark_last_bit) & 1
            container_pixel = np.uint8((container_pixel & 0xFE) | new_bit)

    return container_pixel


def method_extract(watermarked: np.ndarray, original: np.ndarray, method: Method) -> np.uint8:
    """
    Get last bit of watermark for every method
    :param watermarked: data image need to be extracted
    :param original: data original image
    :param method: method was used when embedded
    :return: Last bit of watermark
    """
    container_last_bit = watermarked & 1
    original_container_last_bit = original & 1
    match method:
        case Method.DIRECT:
            return container_last_bit
        case Method.BITWISE_ADD:
            return container_last_bit ^ original_container_last_bit
        case Method.NEGATED_BITWISE_ADD:
            return ~(container_last_bit ^ original_container_last_bit) & 1


def resize_watermark(container: np.ndarray, watermark: np.ndarray) -> np.ndarray:
    """
    Resize watermark like container
    :param container: data container
    :param watermark:  data watermark
    :return: data watermark after resizing
    """
    return cv2.resize(watermark, (container.shape[1], container.shape[0]), interpolation=cv2.INTER_NEAREST)


# Embed image
def embed_watermark(container, watermark, method: Method) -> np.ndarray:
    """
    Embed watermark to image
    :param container: container
    :param watermark: binary image
    :param method: method will be used for embedding
    :return: Image after embedding
    """
    resized_watermark = resize_watermark(container, watermark)

    watermarked = np.copy(container)

    if len(container.shape) == 3:  # RGB image
        for i in range(3):  # Apply watermark to each channel
            for n1 in range(container.shape[0]):
                for n2 in range(container.shape[1]):
                    watermarked[n1, n2, i] = method_embed(container[n1, n2, i], resized_watermark[n1, n2], method)
    else:  # Grayscale image
        for n1 in range(container.shape[0]):
            for n2 in range(container.shape[1]):
                watermarked[n1, n2] = method_embed(container[n1, n2], resized_watermark[n1, n2], method)
    return watermarked


# Extraction function
def extract_watermark(watermarked: np.ndarray, original: np.ndarray,
                      watermark_shape: tuple, method: Method) -> np.ndarray:
    """
    Extract watermark from watermarked image and resize it to original dimensions.
    :param watermarked: watermarked image data
    :param original: original dimensions of the watermark
    :param watermark_shape: size of watermark
    :param method: method was used when embed
    :return: extracted watermark image
    """
    extracted = np.copy(watermarked)

    if len(watermarked.shape) == 3:  # RGB image
        for i in range(3):  # Extract watermark from each channel
            for n1 in range(watermarked.shape[0]):
                for n2 in range(watermarked.shape[1]):
                    extracted[n1, n2] = method_extract(watermarked[n1, n2, i], original[n1, n2, i], method)
    else:  # Grayscale image
        for n1 in range(watermarked.shape[0]):
            for n2 in range(watermarked.shape[1]):
                extracted[n1, n2] = method_extract(watermarked[n1, n2], original[n1, n2], method)
    extracted_resized = cv2.resize(extracted, watermark_shape, interpolation=cv2.INTER_NEAREST)
    _, extracted_resized = cv2.threshold(extracted_resized, 0, 255, cv2.THRESH_BINARY)
    return extracted_resized


# Example usage
if __name__ == "__main__":
    container = cv2.imread('E:/Study/pythonProject/Image/Original/1.png', cv2.IMREAD_UNCHANGED)
    watermark = cv2.imread('E:/Study/pythonProject/Image/Watermark/1.png', cv2.IMREAD_GRAYSCALE)

    _, watermark = cv2.threshold(watermark, 127, 255, cv2.THRESH_BINARY)

    original_watermark_size = (watermark.shape[1], watermark.shape[0])

    method = Method.BITWISE_ADD  # DIRECT,  BITWISE_ADD, NEGATED_BITWISE_ADD

    watermarked_image = embed_watermark(container, watermark, method)
    cv2.imwrite('E:/Study/pythonProject/Image/test.png', watermarked_image)

    extracted_watermark = extract_watermark(watermarked_image, container, original_watermark_size, method)
    cv2.imwrite('E:/Study/pythonProject/Image/extracted.png', extracted_watermark)
