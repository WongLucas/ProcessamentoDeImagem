import numpy as np
import cv2
from scipy.ndimage import convolve

def dilation_manual(image, kernel_size=5):
    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    return cv2.dilate(image, kernel, iterations=1)

def erosion_manual(image, kernel_size=5):
    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    return cv2.erode(image, kernel, iterations=1)

def apply_filter(img_cv, filter_type, display_image, kernel_size=5, scale=1, delta=0):
    if img_cv is None:
        return

    if len(img_cv.shape) == 2 or img_cv.shape[2] == 1:
        img_cv = cv2.cvtColor(img_cv, cv2.COLOR_GRAY2BGR)

    filtered_img = None

    # Garantir que o kernel_size seja ímpar e não maior que 31
    if kernel_size % 2 == 0:
        kernel_size += 1
    if kernel_size > 31:
        kernel_size = 31

    # Filtros de Passa Baixo
    if filter_type == "low_pass_gaussian":
        filtered_img = cv2.GaussianBlur(img_cv, (kernel_size, kernel_size), 0)
    elif filter_type == "low_pass_mean":
        filtered_img = cv2.blur(img_cv, (kernel_size, kernel_size))
    elif filter_type == "low_pass_median":
        filtered_img = cv2.medianBlur(img_cv, kernel_size)

    # Filtros de Passa Alto
    elif filter_type == "high_pass_laplacian":
        gray = cv2.cvtColor(img_cv, cv2.COLOR_BGR2GRAY)
        filtered = cv2.Laplacian(gray, cv2.CV_64F, ksize=kernel_size, scale=scale, delta=delta)
        filtered = cv2.convertScaleAbs(filtered)
        filtered_img = cv2.cvtColor(filtered, cv2.COLOR_GRAY2BGR)
    elif filter_type == "high_pass_sobel":
        gray = cv2.cvtColor(img_cv, cv2.COLOR_BGR2GRAY)
        sobel_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=kernel_size, scale=scale, delta=delta)
        sobel_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=kernel_size, scale=scale, delta=delta)
        filtered = cv2.sqrt(sobel_x**2 + sobel_y**2)
        filtered = cv2.convertScaleAbs(filtered)
        filtered_img = cv2.cvtColor(filtered, cv2.COLOR_GRAY2BGR)
    elif filter_type == "high_pass_roberts":
        gray = cv2.cvtColor(img_cv, cv2.COLOR_BGR2GRAY)
        roberts_x = np.array([[1, 0], [0, -1]])
        roberts_y = np.array([[0, 1], [-1, 0]])
        edges_x = cv2.filter2D(gray, cv2.CV_64F, roberts_x)
        edges_y = cv2.filter2D(gray, cv2.CV_64F, roberts_y)
        filtered = cv2.sqrt(edges_x**2 + edges_y**2)
        filtered = cv2.convertScaleAbs(filtered)
        filtered_img = cv2.cvtColor(filtered, cv2.COLOR_GRAY2BGR)

    if filtered_img is not None:
        display_image(filtered_img, modified=True)  # Exibe a imagem editada

def apply_segmentation(img_cv, segmentation_type, display_image, threshold=127, block_size=11, C=2):
    if img_cv is None:
        return

    segmented_img = None

    # Converte a imagem para escala de cinza usando OpenCV
    gray = cv2.cvtColor(img_cv, cv2.COLOR_BGR2GRAY)

    if segmentation_type == "threshold":
        _, segmented_img = cv2.threshold(gray, threshold, 255, cv2.THRESH_BINARY)
    elif segmentation_type == "adaptive_threshold":
        segmented_img = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, block_size, C)

    if segmented_img is not None:
        display_image(segmented_img, modified=True)  # Exibe a imagem editada
        return segmented_img

def global_threshold(image_array, threshold=127):
    _, binary_image = cv2.threshold(image_array, threshold, 255, cv2.THRESH_BINARY)
    return binary_image

def adaptive_threshold(image_array, block_size=11, C=2):
    return cv2.adaptiveThreshold(image_array, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, block_size, C)

def apply_morphological_operation(img_cv, operation_type, display_image, kernel_size=5, iterations=1):
    if img_cv is None:
        return

    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    morphed_img = None

    if operation_type == "erosion":
        morphed_img = cv2.erode(img_cv, kernel, iterations=iterations)
    elif operation_type == "dilation":
        morphed_img = cv2.dilate(img_cv, kernel, iterations=iterations)
    elif operation_type == "opening":
        morphed_img = cv2.morphologyEx(img_cv, cv2.MORPH_OPEN, kernel, iterations=iterations)
    elif operation_type == "closing":
        morphed_img = cv2.morphologyEx(img_cv, cv2.MORPH_CLOSE, kernel, iterations=iterations)
    elif operation_type == "opening_closing":
        opened = cv2.morphologyEx(img_cv, cv2.MORPH_OPEN, kernel, iterations=iterations)
        morphed_img = cv2.morphologyEx(opened, cv2.MORPH_CLOSE, kernel, iterations=iterations)

    if morphed_img is not None:
        display_image(morphed_img, modified=True)  # Exibe a imagem editada