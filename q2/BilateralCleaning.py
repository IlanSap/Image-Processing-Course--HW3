import cv2
import numpy as np
import matplotlib.pyplot as plt


# Applies bilateral filtering to remove Gaussian noise from a grayscale image.
def clean_Gaussian_noise_bilateral(im, radius, stdSpatial, stdIntensity):
    im_float = im.astype(np.float64)
    # Add border to the original image
    padded_image = np.pad(im_float, ((radius, radius), (radius, radius)), mode='reflect')
    # Initialize as a zero matrix of the same shape as the input image to store the clean image in it
    cleanedImage = np.zeros_like(padded_image, dtype=np.float64)
    # Save the size of the image
    rows, cols = im.shape

    # Generate spatial weights from a Gaussian function
    x, y = np.meshgrid(np.arange(-radius, radius + 1), np.arange(-radius, radius + 1))
    gs = np.exp(-((x ** 2 + y ** 2) / (2 * stdSpatial ** 2)))

    # Apply bilateral filtering
    for i in range(radius, rows + radius):
        for j in range(radius, cols + radius):
            # Extract the window centered at pixel (i, j)
            window = padded_image[i - radius:i + radius + 1, j - radius:j + radius + 1]

            # Compute the intensity weights
            gi = np.exp(-((window - padded_image[i, j]) ** 2) / (2 * stdIntensity ** 2))

            # Combine spatial and intensity weights
            weights = gs * gi

            # Apply weighted averaging
            cleanedImage[i, j] = np.sum(window * weights) / np.sum(weights)

    # Clip and convert the image to uint8
    cleanedImage = np.clip(cleanedImage, 0, 255).astype(np.uint8)
    cleanedImage = cleanedImage[radius:rows + radius, radius:cols + radius]

    return cleanedImage


# Solution #2
def clean_Gaussian_noise_bilateral2(im, radius, stdSpatial, stdIntensity):
    # Add border to the original image
    originalImage = np.pad(im, ((radius, radius), (radius, radius)), mode='reflect').astype(np.float64)
    # Initialize as a zero matrix of the same shape as the input image to store the clean image in it
    cleanedImage = np.zeros(im.shape, dtype=np.float64)
    # Create coordinate grid for the gaussian function
    x, y = np.indices((2 * radius + 1, 2 * radius + 1))
    # Save the size of the image
    rows, cols = cleanedImage.shape

    # Iterates through each pixel of the image and computes the bilateral filter response using the Gaussian kernel in both spatial and intensity domains.
    # Finally, it stores the filtered pixel value in the output image array.
    for i in range(radius, rows + radius):
        for j in range(radius, cols + radius):
            window = originalImage[i - radius:i + radius + 1, j - radius: j + radius + 1]
            gi = np.exp(-((window - originalImage[i, j]) ** 2) / (2 * (stdIntensity ** 2)))
            gs = np.exp(-(((x - i) ** 2) + (y - j) ** 2) / (2 * (stdSpatial ** 2)))
            cleanedImage[i - radius, j - radius] = np.sum(gi * gs * window) / np.sum(gi * gs)
    return cleanedImage.astype(np.uint8)


# Another version of the solution
# def clean_Gaussian_noise_bilateral(im, radius, stdSpatial, stdIntensity):
#     # Define padding size
#     pad_size = radius * 2
#
#     # Create a padded copy of the image
#     padded_im = cv2.copyMakeBorder(im, pad_size, pad_size, pad_size, pad_size, cv2.BORDER_REFLECT).astype(np.float64)
#
#     # Create a new image to store the filtered result
#     filtered_image = np.zeros_like(im, dtype=np.float64)
#
#     rows, cols = im.shape
#
#     # Generate spatial kernel
#     spatial_kernel = np.zeros((2 * radius + 1, 2 * radius + 1))
#     for i in range(-radius, radius + 1):
#         for j in range(-radius, radius + 1):
#             spatial_kernel[i + radius, j + radius] = np.exp(-(i ** 2 + j ** 2) / (2 * stdSpatial ** 2))
#
#     for i in range(rows):
#         for j in range(cols):
#             window = padded_im[i:i + 2 * radius + 1, j:j + 2 * radius + 1]
#
#             # Generate intensity kernel
#             intensity_kernel = np.exp(-((window - padded_im[i + radius, j + radius]) ** 2) / (2 * stdIntensity ** 2))
#
#             # Compute the bilateral filter response
#             bilateral_filter = spatial_kernel * intensity_kernel
#
#             # Normalize the filter and apply it to the window
#             filtered_pixel = np.sum(bilateral_filter * window) / np.sum(bilateral_filter)
#
#             # Store the result
#             filtered_image[i, j] = filtered_pixel
#
#     return filtered_image.astype(np.uint8)


# Read an image file and convert it to grayscale
def read_grayscale_image(file_path):
    return cv2.imread(file_path, cv2.IMREAD_GRAYSCALE).astype(np.float64)


# Saves the image to a file
def save_image(image, file_name):
    cv2.imwrite(file_name, image)


# Displays original and filtered images side by side
def display_images(file_name, original_image, filtered_image):
    plt.suptitle('File Name: ' + file_name, fontsize=14)

    plt.subplot(121)
    plt.imshow(original_image, cmap='gray')
    plt.title('Original Image')

    plt.subplot(122)
    plt.imshow(filtered_image, cmap='gray')
    plt.title('Filtered Image')

    plt.show()


original_image_path = 'balls.jpg'
image = read_grayscale_image(original_image_path)
clean_image = clean_Gaussian_noise_bilateral(image, 30, 40, 20)
display_images(original_image_path, image, clean_image)
save_image(clean_image, 'fixed_' + original_image_path)

original_image_path = 'NoisyGrayImage.png'
image = read_grayscale_image(original_image_path)
height, width = image.shape
diagonal = np.sqrt(height ** 2 + width ** 2)
stdSaptial = 0.02 * diagonal
clean_image = clean_Gaussian_noise_bilateral(image, 15, stdSaptial, 100)
print('stdSaptial:', stdSaptial)
display_images(original_image_path, image, clean_image)
save_image(clean_image, 'fixed_' + original_image_path)

original_image_path = 'taj.jpg'
image = read_grayscale_image(original_image_path)
height, width = image.shape
diagonal = np.sqrt(height ** 2 + width ** 2)
stdSaptial = 0.02 * diagonal
clean_image = clean_Gaussian_noise_bilateral(image, 15, stdSaptial, 30)
print('stdSaptial:', stdSaptial)
display_images(original_image_path, image, clean_image)
save_image(clean_image, 'fixed_' + original_image_path)


