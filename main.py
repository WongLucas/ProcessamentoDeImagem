import tkinter as tk
from tkinter import filedialog
import cv2
from PIL import Image, ImageTk

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
        display_image(img_cv, original=True)  # Exibe a imagem original
        refresh_canvas()

def display_image(img, original=False):
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

    if original:
        original_image_canvas.delete("all")  # Limpa o canvas
        original_image_canvas.image = img_tk  # Mantém a referência viva
        original_image_canvas.create_image(x_offset, y_offset, anchor=tk.NW, image=img_tk)
    else:
        edited_image_canvas.delete("all")  # Limpa o canvas
        edited_image_canvas.image = img_tk
        edited_image_canvas.create_image(x_offset, y_offset, anchor=tk.NW, image=img_tk)

def apply_filter(filter_type):
    global img_cv
    if img_cv is None:
        return

    filtered_img = None

    # Filtros de Passa Baixo
    if filter_type == "low_pass_gaussian":
        filtered_img = cv2.GaussianBlur(img_cv, (15, 15), 0)
    elif filter_type == "low_pass_mean":
        filtered_img = cv2.blur(img_cv, (15, 15))
    
    # Filtros de Passa Alto
    elif filter_type == "high_pass_laplacian":
        gray = cv2.cvtColor(img_cv, cv2.COLOR_BGR2GRAY)
        filtered = cv2.Laplacian(gray, cv2.CV_64F)
        filtered = cv2.convertScaleAbs(filtered)
        filtered_img = cv2.cvtColor(filtered, cv2.COLOR_GRAY2BGR)
    elif filter_type == "high_pass_sobel":
        gray = cv2.cvtColor(img_cv, cv2.COLOR_BGR2GRAY)
        sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=5)
        sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=5)
        sobel = cv2.sqrt(sobelx**2 + sobely**2)
        filtered = cv2.convertScaleAbs(sobel)
        filtered_img = cv2.cvtColor(filtered, cv2.COLOR_GRAY2BGR)

    if filtered_img is not None:
        display_image(filtered_img, original=False)  # Exibe a imagem editada

def refresh_canvas():
    edited_image_canvas.delete("all")  # Limpa o canvas para exibir a nova imagem

# Definindo a GUI
root = tk.Tk()
root.title("Image Processing App")

# Define o tamanho da janela da aplicação
root.geometry("1085x600")
root.config(bg="#2e2e2e")

img_cv = None

# Menu
menu_bar = tk.Menu(root)
root.config(menu=menu_bar)

# File menu
file_menu = tk.Menu(menu_bar, tearoff=0)
menu_bar.add_cascade(label="File", menu=file_menu)
file_menu.add_command(label="Load Image", command=load_image)
file_menu.add_separator()
file_menu.add_command(label="Exit", command=root.quit)

# Filters menu com submenus para Passa Baixo e Passa Alto
filters_menu = tk.Menu(menu_bar, tearoff=0)
menu_bar.add_cascade(label="Filters", menu=filters_menu)

# Submenu Low Pass
low_pass_menu = tk.Menu(filters_menu, tearoff=0)
filters_menu.add_cascade(label="Low Pass Filter", menu=low_pass_menu)
low_pass_menu.add_command(label="Gaussian", command=lambda: apply_filter("low_pass_gaussian"))
low_pass_menu.add_command(label="Mean", command=lambda: apply_filter("low_pass_mean"))

# Submenu High Pass
high_pass_menu = tk.Menu(filters_menu, tearoff=0)
filters_menu.add_cascade(label="High Pass Filter", menu=high_pass_menu)
high_pass_menu.add_command(label="Laplacian", command=lambda: apply_filter("high_pass_laplacian"))
high_pass_menu.add_command(label="Sobel", command=lambda: apply_filter("high_pass_sobel"))

# Cria a canvas para a imagem original com borda (sem background)
original_image_canvas = tk.Canvas(root, width=500, height=500, bg="#2e2e2e", highlightthickness=1, highlightbackground="white")
original_image_canvas.grid(row=0, column=0, padx=20, pady=20)

# Cria a canvas para a imagem editada com borda (sem background)
edited_image_canvas = tk.Canvas(root, width=500, height=500, bg="#2e2e2e", highlightthickness=1, highlightbackground="white")
edited_image_canvas.grid(row=0, column=1, padx=20, pady=20)

root.mainloop()
