import os

import cv2
from matplotlib import pyplot as plt
from scipy.ndimage import uniform_filter, gaussian_filter
from skimage import io, util, color, filters
import numpy as np


def display_Result_SideBySide(original, result, title):
    # Ensure images have the same dimensions
    height = min(original.shape[0], result.shape[0])
    width = min(original.shape[1], result.shape[1])

    # Resize images to the same dimensions
    original_resized = cv2.resize(original, (width, height))
    result_resized = cv2.resize(result, (width, height))

    # Convert images to the same data type
    original_resized = cv2.cvtColor(original_resized, cv2.COLOR_BGR2RGB)
    result_resized = cv2.cvtColor(result_resized, cv2.COLOR_BGR2RGB)

    # Concatenate images side by side
    side_by_side = np.hstack((original_resized, result_resized))

    # Display the concatenated image
    plt.figure(figsize=(10, 5))
    plt.imshow(side_by_side)
    plt.title(title)
    plt.axis('off')
    plt.show()

# def display_Result_SideBySide(original, result, title):
#     side_by_side = cv2.hconcat([original, result])
#     cv2.imshow(title, side_by_side)
#     cv2.waitKey(0)
#     cv2.destroyAllWindows()


def unsharp_mask(image, sigma=0.5, strength=1):
    blurred = gaussian_filter(image, sigma=sigma)
    sharpened = image + strength * (image - blurred)
    return np.clip(sharpened, 0, 255).astype(np.uint8)


def histogramEqualization(image):
    equalized_image = cv2.equalizeHist(image)
    return equalized_image


def gammaCorrection(image):
    gamma = 1.5

    # Perform gamma correction
    corrected_image = np.power(image / 255.0, gamma)
    corrected_image = (corrected_image * 255).astype(np.uint8)

    return corrected_image


def apply_gaussian_blur(image, kernel_size=(5, 5), sigma=1.0):
    # Apply Gaussian blur
    blurred_image = cv2.GaussianBlur(image, kernel_size, sigma)
    return blurred_image


def remove_Gaussian_noise(image):
    noisy_image = image + np.random.normal(scale=25, size=image.shape).astype(np.uint8)

    # Apply Gaussian blur to remove Gaussian noise
    kernel_size = (5, 5)  # Adjust the kernel size as needed
    sigma = 1.0  # Adjust the sigma value as needed
    denoised_image = apply_gaussian_blur(noisy_image, kernel_size, sigma)
    return denoised_image


def bilateral_smoothing(image, d=4, sigmaColor=100, sigmaSpace=100):
    smoothed_image = cv2.bilateralFilter(image, d, sigmaColor, sigmaSpace)
    return smoothed_image


def edge_enhancement(image, alpha=2):
    # Apply Sobel filter for gradient-based edge detection
    sobel_x = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=3)
    sobel_y = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=3)

    # Combine the gradients to get the magnitude
    gradient_magnitude = np.sqrt(sobel_x ** 2 + sobel_y ** 2)

    # Normalize the magnitude to [0, 255]
    gradient_magnitude = np.clip((gradient_magnitude / gradient_magnitude.max()) * 255, 0, 255).astype(np.uint8)

    # Add the normalized gradient magnitude to the original image for enhancement
    enhanced_image = cv2.addWeighted(image, 1, gradient_magnitude, 1, 1)

    return enhanced_image


def mean_filter(image, kernel_size):
    # Create a mean filter kernel
    kernel = np.ones((kernel_size, kernel_size), dtype=np.float32) / (kernel_size * kernel_size)

    # Apply convolution using OpenCV's filter2D function
    filtered_image = cv2.filter2D(image, -1, kernel)

    return filtered_image


# Apply mean filter to remove noise
def mean_filter_b(image):
    # Load the noisy images
    noisy_images = np.load(os.path.join('noised_images.npy'))
    # Calculate the average across all images to get the cleaned image
    cleaned_image = np.mean(noisy_images, axis=0)
    # Convert to uint8 data type
    cleaned_image = np.uint8(cleaned_image)
    # Save the cleaned image
    #cv2.imwrite(os.path.join('fixed_image_b.jpg'), cleaned_image)
    return cleaned_image


if __name__ == '__main__':
    # load the image 'broken'
    noisedVegetables = cv2.imread('broken.jpg')

    remove_salt_and_pepper = filters.median(noisedVegetables)
    display_Result_SideBySide(noisedVegetables, remove_salt_and_pepper,'with VS without salt_and_pepper')

    # Apply mean filter to remove noise
    kernel_size = 3 # Adjust the kernel size as needed
    mean_filtered_image = mean_filter(remove_salt_and_pepper, kernel_size)
    # mean_filtered_image = uniform_filter(remove_salt_and_pepper, size=kernel_size)
    display_Result_SideBySide(remove_salt_and_pepper, mean_filtered_image, 'before & after mean filter')

    # Apply bilateral smoothing
    smoothed_image = bilateral_smoothing(mean_filtered_image)
    display_Result_SideBySide(mean_filtered_image, smoothed_image, 'mean_filtered_image VS smoothed image')

    cv2.imwrite(os.path.join('fixed_image.jpg'), smoothed_image)

    # Apply mean filter to remove noise - section b
    mean_filtered_image_b = mean_filter_b(noisedVegetables)
    display_Result_SideBySide(noisedVegetables, mean_filtered_image_b, 'broken image VS broken image after section b')
    cv2.imwrite(os.path.join('fixed_image_b.jpg'), mean_filtered_image_b)