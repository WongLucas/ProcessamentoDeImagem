import numpy as np
import cv2
from scipy.ndimage import convolve

def dilation_manual(image, kernel_size=5):
    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    eroded_image = np.zeros_like(image)
    offset = kernel_size // 2

    for i in range(offset, image.shape[0] - offset):
        for j in range(offset, image.shape[1] - offset):
            neighborhood = image[i - offset:i + offset + 1, j - offset:j + offset + 1]
            if np.all(neighborhood == 255):
                eroded_image[i, j] = 255
            else:
                eroded_image[i, j] = 0

    return eroded_image

def erosion_manual(image, kernel_size=5):
    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    dilated_image = np.zeros_like(image)
    offset = kernel_size // 2

    for i in range(offset, image.shape[0] - offset):
        for j in range(offset, image.shape[1] - offset):
            neighborhood = image[i - offset:i + offset + 1, j - offset + offset + 1]
            if np.any(neighborhood == 255):
                dilated_image[i, j] = 255
            else:
                dilated_image[i, j] = 0

    return dilated_image

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
        kernel = cv2.getGaussianKernel(kernel_size, -1)
        kernel = kernel @ kernel.T
        filtered_img = np.zeros_like(img_cv)
        for i in range(3):  # Aplica o filtro a cada canal
            filtered_img[:, :, i] = convolve(img_cv[:, :, i], kernel)

    elif filter_type == "low_pass_mean":
        kernel = np.ones((kernel_size, kernel_size)) / (kernel_size * kernel_size)
        filtered_img = np.zeros_like(img_cv)
        for i in range(3):  # Aplica o filtro a cada canal
            filtered_img[:, :, i] = convolve(img_cv[:, :, i], kernel)

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
        segmented_img = global_threshold(gray, threshold=threshold)
    elif segmentation_type == "adaptive_threshold":
        segmented_img = adaptive_threshold(gray, block_size=block_size, C=C)

    if segmented_img is not None:
        display_image(segmented_img, modified=True)  # Exibe a imagem editada
        return segmented_img

def global_threshold(image_array, threshold=127):
    # Aplique o threshold a cada pixel
    binary_image = (image_array > threshold) * 255
    return binary_image.astype(np.uint8)

def adaptive_threshold(image_array, block_size=11, C=2):
    # Obtém as dimensões da imagem
    height, width = image_array.shape

    # Cria uma nova matriz para armazenar a imagem binária
    binary_image = np.zeros_like(image_array)

    # Define o offset para calcular a vizinhança
    offset = block_size // 2

    # Itera por cada pixel na imagem
    for i in range(offset, height - offset):
        for j in range(offset, width - offset):
            # Calcula a média dos pixels na vizinhança
            local_mean = np.mean(image_array[i-offset:i+offset+1, j-offset:j+offset+1])
            # Aplica o threshold adaptativo
            if image_array[i, j] > local_mean - C:
                binary_image[i, j] = 255
            else:
                binary_image[i, j] = 0

    return binary_image.astype(np.uint8)

def apply_morphological_operation(img_cv, operation_type, display_image, kernel_size=5, iterations=1):
    if img_cv is None:
        return

    # As operações morfológicas manuais
    morphed_img = None

    if operation_type == "erosion":
        morphed_img = erosion_manual(img_cv, kernel_size=kernel_size)
    elif operation_type == "dilation":
        morphed_img = dilation_manual(img_cv, kernel_size=kernel_size)
    elif operation_type == "opening":
        eroded_image = erosion_manual(img_cv, kernel_size=kernel_size)
        morphed_img = dilation_manual(eroded_image, kernel_size=kernel_size)
    elif operation_type == "closing":
        dilated_image = dilation_manual(img_cv, kernel_size=kernel_size)
        morphed_img = erosion_manual(dilated_image, kernel_size=kernel_size)
    elif operation_type == "opening_closing":
        eroded_image = erosion_manual(img_cv, kernel_size=kernel_size)
        opened_image = dilation_manual(eroded_image, kernel_size=kernel_size)
        dilated_image = dilation_manual(opened_image, kernel_size=kernel_size)
        morphed_img = erosion_manual(dilated_image, kernel_size=kernel_size)

    if morphed_img is not None:
        display_image(morphed_img, modified=True)  # Exibe a imagem editada