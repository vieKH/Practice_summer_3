import numpy as np
import cv2

# Embedding methods
def direct_replacement(container_pixel, watermark_pixel):
    return watermark_pixel


def bitwise_addition(container_pixel, watermark_pixel):
    return container_pixel ^ watermark_pixel


def negated_bitwise_addition(container_pixel, watermark_pixel):
    return ~(container_pixel ^ watermark_pixel) & 0xFF


def resize_watermark(container, watermark):
    return cv2.resize(watermark, (container.shape[1], container.shape[0]), interpolation=cv2.INTER_NEAREST)


def embed_watermark(container, watermark, method):
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
                    watermarked[n1, n2, i] = method(container[n1, n2, i], resized_watermark[n1, n2])
    else:  # Grayscale image
        for n1 in range(container.shape[0]):
            for n2 in range(container.shape[1]):
                watermarked[n1, n2] = method(container[n1, n2], resized_watermark[n1, n2])
    return watermarked


# Extraction function
def extract_watermark(watermarked, container, original_watermark_size, method):
    """
    Extract watermark image from 1 image
    :param watermarked: Image need to be extracted
    :param container: container
    :param original_watermark_size: size watermark image
    :param method: method was used when embed
    :return: watermark image
    """
    extracted = np.copy(watermarked)

    if method == direct_replacement:
        pass
    elif method == bitwise_addition:
        for n1 in range(watermarked.shape[0]):
            for n2 in range(watermarked.shape[1]):
                if len(watermarked.shape) == 3:  # RGB image
                    for i in range(3):
                        extracted[n1, n2, i] = watermarked[n1, n2, i] ^ container[n1, n2, i]
                else:
                    extracted[n1, n2] = watermarked[n1, n2] ^ container[n1, n2]
    elif method == negated_bitwise_addition:
        for n1 in range(watermarked.shape[0]):
            for n2 in range(watermarked.shape[1]):
                if len(watermarked.shape) == 3:  # RGB image
                    for i in range(3):
                        extracted[n1, n2, i] = ~(watermarked[n1, n2, i] ^ container[n1, n2, i]) & 0xFF
                else:
                    extracted[n1, n2] = ~(watermarked[n1, n2] ^ container[n1, n2]) & 0xFF

    extracted_resized = cv2.resize(extracted, original_watermark_size, interpolation=cv2.INTER_NEAREST)
    return extracted_resized


# Example usage
if __name__ == "__main__":
    container = cv2.imread('E:/Study/pythonProject/Image/Original/1.png', cv2.IMREAD_UNCHANGED)
    watermark = cv2.imread('E:/Study/pythonProject/Image/Watermark/1.png', cv2.IMREAD_GRAYSCALE)

    # Ensure the watermark is binary
    _, watermark = cv2.threshold(watermark, 127, 255, cv2.THRESH_BINARY)

    original_watermark_size = (watermark.shape[1], watermark.shape[0])

    method = negated_bitwise_addition  # Can be direct_replacement, bitwise_addition, or negated_bitwise_addition

    watermarked_image = embed_watermark(container, watermark, method)
    cv2.imwrite('E:/Study/pythonProject/Image/test.png', watermarked_image)

    extracted_watermark = extract_watermark(watermarked_image, container, original_watermark_size, method)
    cv2.imwrite('E:/Study/pythonProject/Image/extracted.png', extracted_watermark)
