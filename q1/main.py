import cv2
import numpy as np
import os

def calculate_mse(image1, image2):
    # Ensure the images have the same shape
    if image1.shape != image2.shape:
        image2 = cv2.resize(image2, (image1.shape[1], image1.shape[0]))

    # Compute MSE
    mse = np.sum((image1 - image2)**2) / float(image1.size)
    print("The MSE is: ",mse)


def apply_mean_filter(image, kernel_size):

    # Apply mean filter
    smoothed_image = cv2.blur(image, kernel_size)
    cv2.imshow('Mean Filter', smoothed_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    return smoothed_image


def apply_sharp(image):
    # Define a sharpening kernel
    # apply Edge Enhancement by Filtering (sum of the kernel numbers equal to 1
    sharpening_kernel = np.array([[0, -1, 0], [-1, 6, -1], [0, -1, 0]]) * 0.5

    # Apply convolution using cv2.filter2D
    sharpened_image = cv2.filter2D(image, -1, sharpening_kernel)

    cv2.imshow('Sharpened Image', sharpened_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    return sharpened_image


def apply_cyclic_pattern(image):

    filtered = np.roll(image, image.shape[0] // 2, axis=0)

    cv2.imshow('Upward Padding Result', filtered)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    return filtered


def apply_convolution_with_Gaussian_Mask(image):

    # Define a custom Laplacian kernel (example: 3x3 kernel)
    blured_image = cv2.GaussianBlur(image,(9,9),10)
    details_image =cv2.subtract(image, blured_image)
    enhanced_details = cv2.addWeighted(image, 1, blured_image, -1, 128)
    enhanced_details = cv2.normalize(enhanced_details, None, 0, 255, cv2.NORM_MINMAX)

    cv2.imshow('apply_convolution_with_Gaussian_Mask', enhanced_details)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    return enhanced_details


def apply_custom_laplacian(image):
    laplacian_kernel = np.array([
        [0, -1, 0],
        [-1, 4, -1],
        [0, -1, 0]
    ])

    # Apply convolution using OpenCV's filter2D function
    filtered_image = cv2.filter2D(image, -1, laplacian_kernel)
    laplacian_image = np.uint8(np.absolute(filtered_image))

    cv2.imshow('Laplacian Filtered Image', laplacian_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    return filtered_image


def apply_gaussian_blur(image, kernel_size, sigma):

    blurred_image = cv2.GaussianBlur(image, kernel_size, sigma)

    return blurred_image


def apply_median_filter(image, window_size):
    
    result = cv2.medianBlur(image, window_size)
    cv2.imshow('gaussian Mask Convolution', result)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    return result


def apply_delta_mask(image):
    laplacian_kernel = np.array([[0, 0, 0],
                                 [0, 1, 0],
                                 [0, 0, 0]], dtype=np.float32)

    # Apply convolution using cv2.filter2D with the Laplacian kernel
    filtered_image = cv2.filter2D(image, cv2.CV_64F, laplacian_kernel)

    # Normalize the result to be in the range [0, 255]
    filtered_image = cv2.normalize(filtered_image, None, 0, 255, cv2.NORM_MINMAX)

    # Convert the result to uint8 format
    filtered_image = np.uint8(filtered_image)

    cv2.imshow('Delta Mask Filtered Image', filtered_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    return filtered_image

# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    realImagePath = os.path.join('images', '1.jpg')
    realImage = cv2.imread(realImagePath, cv2.IMREAD_GRAYSCALE)

    image_1 = cv2.imread(os.path.join('images', 'image_1.jpg'), cv2.IMREAD_GRAYSCALE)
    image_2 = cv2.imread(os.path.join('images', 'image_2.jpg'), cv2.IMREAD_GRAYSCALE)
    image_3 = cv2.imread(os.path.join('images', 'image_3.jpg'), cv2.IMREAD_GRAYSCALE)
    image_4 = cv2.imread(os.path.join('images', 'image_4.jpg'), cv2.IMREAD_GRAYSCALE)
    image_5 = cv2.imread(os.path.join('images', 'image_5.jpg'), cv2.IMREAD_GRAYSCALE)
    image_6 = cv2.imread(os.path.join('images', 'image_6.jpg'), cv2.IMREAD_GRAYSCALE)
    image_7 = cv2.imread(os.path.join('images', 'image_7.jpg'), cv2.IMREAD_GRAYSCALE)
    image_8 = cv2.imread(os.path.join('images', 'image_8.jpg'), cv2.IMREAD_GRAYSCALE)
    image_9 = cv2.imread(os.path.join('images', 'image_9.jpg'), cv2.IMREAD_GRAYSCALE)

    # compare to image_1
    verticalMaskConv = apply_mean_filter(realImage, (1180, 1))
    calculate_mse(image_1, verticalMaskConv)
    cv2.imwrite(os.path.join('recreation_images', 'image_1.jpg'), verticalMaskConv)

    # compare to image_2
    meanFilteredImage = apply_mean_filter(realImage, (11,11))
    calculate_mse(image_2, meanFilteredImage)
    cv2.imwrite(os.path.join('recreation_images', 'image_2.jpg'), meanFilteredImage)

    # compare to image_3
    medianFilterImage = apply_median_filter(realImage, 9)
    calculate_mse(image_3, medianFilterImage)
    cv2.imwrite(os.path.join('recreation_images', 'image_3.jpg'), medianFilterImage)

    # compare to image_4
    verticalMaskConv = apply_mean_filter(realImage, (2, 13))
    calculate_mse(image_4, verticalMaskConv)
    cv2.imwrite(os.path.join('recreation_images', 'image_4.jpg'), verticalMaskConv)

    # compare to image_5
    gausMaskFilter = apply_convolution_with_Gaussian_Mask(realImage)
    calculate_mse(image_5, gausMaskFilter)
    cv2.imwrite(os.path.join('recreation_images', 'image_5.jpg'), gausMaskFilter)

    # compare to image_6
    laplacianFilterImage = apply_custom_laplacian(realImage)
    calculate_mse(image_6, laplacianFilterImage)
    cv2.imwrite(os.path.join('recreation_images', 'image_6.jpg'), laplacianFilterImage)

    # compare to image_7
    cyclicImage = apply_cyclic_pattern(realImage)
    calculate_mse(image_7, cyclicImage)
    cv2.imwrite(os.path.join('recreation_images', 'image_7.jpg'), cyclicImage)

    # compare to image_8
    deltaMaskFilterImage = apply_delta_mask(realImage)
    calculate_mse(image_8, deltaMaskFilterImage)
    cv2.imwrite(os.path.join('recreation_images', 'image_8.jpg'), deltaMaskFilterImage)

    # compare to image_9
    sharpenImage = apply_sharp(realImage)
    calculate_mse(image_9, sharpenImage)
    cv2.imwrite(os.path.join('recreation_images', 'image_9.jpg'), sharpenImage)


