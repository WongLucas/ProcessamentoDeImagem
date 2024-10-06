import cv2
import numpy as np

def apply_filter(img_cv, filter_type, display_image):
    if img_cv is None:
        return

    filtered_img = None

    # Filtros de Passa Baixo
    if filter_type == "low_pass_gaussian":
        filtered_img = cv2.GaussianBlur(img_cv, (15, 15), 0)
    elif filter_type == "low_pass_mean":
        filtered_img = cv2.blur(img_cv, (15, 15))
    elif filter_type == "low_pass_median":
        filtered_img = cv2.medianBlur(img_cv, 15)
    
    # Filtros de Passa Alto
    elif filter_type == "high_pass_laplacian":
        gray = cv2.cvtColor(img_cv, cv2.COLOR_BGR2GRAY)
        filtered = cv2.Laplacian(gray, cv2.CV_64F)
        filtered = cv2.convertScaleAbs(filtered)
        filtered_img = cv2.cvtColor(filtered, cv2.COLOR_GRAY2BGR)
    elif filter_type == "high_pass_sobel":
        gray = cv2.cvtColor(img_cv, cv2.COLOR_BGR2GRAY)

        # Kernel Sobel para detectar bordas na direção X
        sobel_x = np.array([[-1, -2, -1],
                             [0, 0, 0],
                             [1, 2, 1]])

        # Kernel Sobel para detectar bordas na direção Y
        sobel_y = np.array([[-1, 0, 1],
                             [-2, 0, 2],
                             [-1, 0, 1]])

        # Aplicar os filtros Sobel
        edges_x = cv2.filter2D(gray, cv2.CV_64F, sobel_x)
        edges_y = cv2.filter2D(gray, cv2.CV_64F, sobel_y)

        # Combinar as bordas X e Y
        filtered_img = cv2.sqrt(edges_x**2 + edges_y**2)
        filtered_img = cv2.convertScaleAbs(filtered_img)
        filtered_img = cv2.cvtColor(filtered_img, cv2.COLOR_GRAY2BGR)
        
    elif filter_type == "high_pass_roberts":
        gray = cv2.cvtColor(img_cv, cv2.COLOR_BGR2GRAY)

        # Kernels de Roberts
        roberts_x = np.array([[1, 0],
                               [0, -1]])
        roberts_y = np.array([[0, 1],
                               [-1, 0]])

        # Aplicar o filtro de Roberts
        edges_x = cv2.filter2D(gray, cv2.CV_64F, roberts_x)
        edges_y = cv2.filter2D(gray, cv2.CV_64F, roberts_y)
        filtered_img = cv2.sqrt(edges_x**2 + edges_y**2)
        filtered_img = cv2.convertScaleAbs(filtered_img)
        filtered_img = cv2.cvtColor(filtered_img, cv2.COLOR_GRAY2BGR)

    if filtered_img is not None:
        display_image(filtered_img, original=False)  # Exibe a imagem editada