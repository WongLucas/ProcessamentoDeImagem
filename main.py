import tkinter as tk
from tkinter import filedialog
import cv2
from PIL import Image, ImageTk
import numpy as np
from utils import apply_filter, apply_morphological_operation, apply_segmentation

def load_image():
    global img_cv
    file_path = filedialog.askopenfilename(
        filetypes=[
            ("Image Files", "*.png;*.jpg;*.jpeg;*.bmp;*.tiff"),
            ("All Files", "*.*")
        ]
    )
    if file_path:
        img_cv = cv2.imread(file_path)
        display_image(img_cv, modified=False)  # Exibe a imagem original

def display_image(img, modified=False):
    global img_cv, modified_img_cv
    if modified:
        modified_img_cv = img  # Atualiza a imagem modificada
    else:
        img_cv = img  # Atualiza a imagem atual

    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_pil = Image.fromarray(img_rgb)
    
    # Redimensiona a imagem para caber no canvas se for muito grande
    max_size = 500
    img_pil.thumbnail((max_size, max_size))  # Mantém a proporção
    img_tk = ImageTk.PhotoImage(img_pil)

    # Calcula a posição para centralizar a imagem dentro do canvas se for menor
    canvas_width, canvas_height = max_size, max_size
    x_offset = (canvas_width - img_pil.width) // 2
    y_offset = (canvas_height - img_pil.height) // 2

    if modified:
        modified_image_canvas.delete("all")  # Limpa o canvas
        modified_image_canvas.image = img_tk  # Mantém a referência viva
        modified_image_canvas.create_image(x_offset, y_offset, anchor=tk.NW, image=img_tk)
    else:
        original_image_canvas.delete("all")  # Limpa o canvas
        original_image_canvas.image = img_tk  # Mantém a referência viva
        original_image_canvas.create_image(x_offset, y_offset, anchor=tk.NW, image=img_tk)

def approve_change():
    global img_cv, modified_img_cv
    if modified_img_cv is not None:
        img_cv = modified_img_cv
        display_image(img_cv, modified=False)  # Atualiza a imagem atual com a modificada

def refresh_canvas():
    original_image_canvas.delete("all")  # Limpa o canvas para exibir a nova imagem
    modified_image_canvas.delete("all")  # Limpa o canvas para exibir a nova imagem

def update_filter(filter_type):
    global current_filter
    current_filter = filter_type
    
    # Oculta todos os sliders inicialmente
    kernel_size_slider.grid_remove()
    scale_slider.grid_remove()
    delta_slider.grid_remove()
    threshold_slider.grid_remove()
    block_size_slider.grid_remove()
    C_slider.grid_remove()
    iterations_slider.grid_remove()

    # Exibe apenas os sliders relevantes para o filtro selecionado
    if filter_type in ["low_pass_gaussian", "low_pass_mean", "low_pass_median"]:
        kernel_size_slider.grid(row=0, column=0, padx=5, pady=10, sticky="ew")
        kernel_size = kernel_size_slider.get()
        apply_filter(img_cv, filter_type, display_image, kernel_size)
    elif filter_type in ["high_pass_laplacian", "high_pass_sobel", "high_pass_roberts"]:
        kernel_size_slider.grid(row=0, column=0, padx=5, pady=10, sticky="ew")
        scale_slider.grid(row=1, column=0, padx=5, pady=10, sticky="ew")
        delta_slider.grid(row=2, column=0, padx=5, pady=10, sticky="ew")
        kernel_size = kernel_size_slider.get()
        scale = scale_slider.get()
        delta = delta_slider.get()
        apply_filter(img_cv, filter_type, display_image, kernel_size, scale, delta)
    elif filter_type in ["threshold", "adaptive_threshold"]:
        if filter_type == "threshold":
            threshold_slider.grid(row=3, column=0, padx=5, pady=10, sticky="ew")
            threshold = threshold_slider.get()
            apply_segmentation(img_cv, filter_type, display_image, threshold)
        elif filter_type == "adaptive_threshold":
            block_size_slider.grid(row=4, column=0, padx=5, pady=10, sticky="ew")
            C_slider.grid(row=5, column=0, padx=5, pady=10, sticky="ew")
            block_size = block_size_slider.get()
            C = C_slider.get()
            apply_segmentation(img_cv, filter_type, display_image, block_size=block_size, C=C)
    elif filter_type in ["erosion", "dilation", "opening", "closing", "opening_closing"]:
        kernel_size_slider.grid(row=0, column=0, padx=5, pady=10, sticky="ew")
        iterations_slider.grid(row=6, column=0, padx=5, pady=10, sticky="ew")
        kernel_size = kernel_size_slider.get()
        iterations = iterations_slider.get()
        apply_morphological_operation(img_cv, filter_type, display_image, kernel_size, iterations)

# Definindo a GUI
root = tk.Tk()
root.title("Image Processing App")

# Define o tamanho da janela da aplicação
root.geometry("1300x700")
root.config(bg="#2e2e2e")

img_cv = None
modified_img_cv = None
current_filter = "low_pass_gaussian"  # Filtro padrão inicial

# Menu
menu_bar = tk.Menu(root)
root.config(menu=menu_bar)

# File menu
file_menu = tk.Menu(menu_bar, tearoff=0)
menu_bar.add_cascade(label="File", menu=file_menu)
file_menu.add_command(label="Load Image", command=load_image)
file_menu.add_separator()
file_menu.add_command(label="Exit", command=root.quit)

# Low Pass Filter menu
low_pass_menu = tk.Menu(menu_bar, tearoff=0)
menu_bar.add_cascade(label="Low Pass Filter", menu=low_pass_menu)
low_pass_menu.add_command(label="Gaussian", command=lambda: update_filter("low_pass_gaussian"))
low_pass_menu.add_command(label="Mean", command=lambda: update_filter("low_pass_mean"))
low_pass_menu.add_command(label="Median", command=lambda: update_filter("low_pass_median"))

# High Pass Filter menu
high_pass_menu = tk.Menu(menu_bar, tearoff=0)
menu_bar.add_cascade(label="High Pass Filter", menu=high_pass_menu)
high_pass_menu.add_command(label="Laplacian", command=lambda: update_filter("high_pass_laplacian"))
high_pass_menu.add_command(label="Sobel", command=lambda: update_filter("high_pass_sobel"))
high_pass_menu.add_command(label="Roberts", command=lambda: update_filter("high_pass_roberts"))

# Segmentation menu
segmentation_menu = tk.Menu(menu_bar, tearoff=0)
menu_bar.add_cascade(label="Segmentation", menu=segmentation_menu)
segmentation_menu.add_command(label="Thresholding", command=lambda: update_filter("threshold"))
segmentation_menu.add_command(label="Adaptive Thresholding", command=lambda: update_filter("adaptive_threshold"))

# Morphological Operations menu
morph_menu = tk.Menu(menu_bar, tearoff=0)
menu_bar.add_cascade(label="Morphological Operations", menu=morph_menu)
morph_menu.add_command(label="Erosion", command=lambda: update_filter("erosion"))
morph_menu.add_command(label="Dilation", command=lambda: update_filter("dilation"))
morph_menu.add_command(label="Opening", command=lambda: update_filter("opening"))
morph_menu.add_command(label="Closing", command=lambda: update_filter("closing"))
morph_menu.add_command(label="Opening and Closing", command=lambda: update_filter("opening_closing"))

# Cria a canvas para a imagem original com borda (sem background)
original_image_canvas = tk.Canvas(root, width=550, height=550, bg="#2e2e2e", highlightthickness=1, highlightbackground="white")
original_image_canvas.grid(row=0, column=0, padx=20, pady=20)

# Cria a canvas para a imagem modificada com borda (sem background)
modified_image_canvas = tk.Canvas(root, width=550, height=550, bg="#2e2e2e", highlightthickness=1, highlightbackground="white")
modified_image_canvas.grid(row=0, column=1, padx=20, pady=20)

# Frame para os sliders
slider_frame = tk.Frame(root, bg="#2e2e2e")
slider_frame.grid(row=0, column=3, columnspan=2, pady=10)

# Botão para aprovar a mudança
approve_button = tk.Button(root, text="Approve Change", command=approve_change)
approve_button.grid(row=2, column=0, columnspan=2, pady=10)

# Controle deslizante para ajustar o tamanho do kernel
kernel_size_slider = tk.Scale(slider_frame, from_=3, to=9, orient=tk.HORIZONTAL, label="Kernel Size", command=lambda x: update_filter(current_filter))
kernel_size_slider.set(5)
kernel_size_slider.grid(row=0, column=0, padx=5, pady=10, sticky="ew")
kernel_size_slider.grid_remove()  # Inicialmente oculto

# Controle deslizante para ajustar a escala
scale_slider = tk.Scale(slider_frame, from_=1, to=10, orient=tk.HORIZONTAL, label="Scale", command=lambda x: update_filter(current_filter))
scale_slider.set(1)
scale_slider.grid(row=1, column=0, padx=5, pady=10, sticky="ew")
scale_slider.grid_remove()  # Inicialmente oculto

# Controle deslizante para ajustar o delta
delta_slider = tk.Scale(slider_frame, from_=0, to=100, orient=tk.HORIZONTAL, label="Delta", command=lambda x: update_filter(current_filter))
delta_slider.set(0)
delta_slider.grid(row=2, column=0, padx=5, pady=10, sticky="ew")
delta_slider.grid_remove()  # Inicialmente oculto

# Controle deslizante para ajustar o threshold
threshold_slider = tk.Scale(slider_frame, from_=0, to=255, orient=tk.HORIZONTAL, label="Threshold", command=lambda x: update_filter(current_filter))
threshold_slider.set(127)
threshold_slider.grid(row=3, column=0, padx=5, pady=10, sticky="ew")
threshold_slider.grid_remove()  # Inicialmente oculto

# Controle deslizante para ajustar o block size
block_size_slider = tk.Scale(slider_frame, from_=3, to=31, orient=tk.HORIZONTAL, label="Block Size", command=lambda x: update_filter(current_filter))
block_size_slider.set(11)
block_size_slider.grid(row=4, column=0, padx=5, pady=10, sticky="ew")
block_size_slider.grid_remove()  # Inicialmente oculto

# Controle deslizante para ajustar o C
C_slider = tk.Scale(slider_frame, from_=-10, to=10, orient=tk.HORIZONTAL, label="C", command=lambda x: update_filter(current_filter))
C_slider.set(2)
C_slider.grid(row=5, column=0, padx=5, pady=10, sticky="ew")
C_slider.grid_remove()  # Inicialmente oculto

# Controle deslizante para ajustar o número de iterações
iterations_slider = tk.Scale(slider_frame, from_=1, to=10, orient=tk.HORIZONTAL, label="Iterations", command=lambda x: update_filter(current_filter))
iterations_slider.set(1)
iterations_slider.grid(row=6, column=0, padx=5, pady=10, sticky="ew")
iterations_slider.grid_remove()  # Inicialmente oculto

root.mainloop()