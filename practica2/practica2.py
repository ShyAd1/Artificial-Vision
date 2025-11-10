import cv2
import numpy as np
import matplotlib.pyplot as plt
import random
import tkinter as tk
from tkinter import filedialog, ttk, messagebox
from PIL import Image, ImageTk

# ---------------- FUNCIONES ----------------
def agregar_ruido_sal_pimienta(imagen, prob):
    salida = imagen.copy()
    alto, ancho = salida.shape
    for i in range(alto):
        for j in range(ancho):
            r = random.random()
            if r < prob / 2:
                salida[i, j] = 0
            elif r < prob:
                salida[i, j] = 255
    return salida

def filtro_mediana(imagen, k=3):
    alto, ancho = imagen.shape
    borde = k // 2
    salida = np.zeros_like(imagen)
    for i in range(borde, alto - borde):
        for j in range(borde, ancho - borde):
            ventana = imagen[i - borde : i + borde + 1, j - borde : j + borde + 1]
            salida[i, j] = np.median(ventana)
    return salida

def aplicar_filtro():
    prob = slider_ruido.get() / 100.0
    imagen_ruido = agregar_ruido_sal_pimienta(imagen_gray, prob)
    imagen_filtrada = filtro_mediana(imagen_ruido, k=3)

    plt.figure(figsize=(12, 6))
    plt.subplot(1, 3, 1)
    plt.imshow(imagen_gray, cmap="gray")
    plt.title("Imagen en gris")
    plt.axis("off")

    plt.subplot(1, 3, 2)
    plt.imshow(imagen_ruido, cmap="gray")
    plt.title(f"Ruido Sal y Pimienta (p={prob:.2f})")
    plt.axis("off")

    plt.subplot(1, 3, 3)
    plt.imshow(imagen_filtrada, cmap="gray")
    plt.title("Filtro Mediana")
    plt.axis("off")

    plt.tight_layout()
    plt.show()

    # Preguntar si quiere otra imagen
    respuesta = messagebox.askyesno("Reintentar", "¿Desea probar con otra imagen?")
    ventana.destroy()
    if respuesta:
        iniciar_programa()

# ---------------- FUNCIÓN PRINCIPAL ----------------
def iniciar_programa():
    global imagen_gray, ventana, slider_ruido

    # Selección de archivo
    root = tk.Tk()
    root.withdraw()
    ruta = filedialog.askopenfilename(
        title="Seleccionar imagen",
        filetypes=[("Archivos de imagen", "*.png;*.jpg;*.jpeg;*.bmp;*.tif")]
    )
    root.destroy()

    if not ruta:
        print("No seleccionaste ninguna imagen. Programa finalizado.")
        return

    imagen = cv2.imread(ruta, cv2.IMREAD_COLOR)
    if imagen is None:
        print("No se pudo cargar la imagen.")
        return

    imagen_gray = cv2.cvtColor(imagen, cv2.COLOR_BGR2GRAY)

    # Crear ventana principal
    ventana = tk.Tk()
    ventana.title("Filtro de Mediana")
    ventana.geometry("500x300")

    # Convertir imagen a formato Tkinter DESPUÉS de crear ventana
    img_rgb = cv2.cvtColor(imagen, cv2.COLOR_BGR2RGB)
    img_pil = Image.fromarray(img_rgb).resize((200, 200))
    img_tk = ImageTk.PhotoImage(img_pil)

    # Mostrar imagen original
    lbl_img = tk.Label(ventana, image=img_tk)
    lbl_img.image = img_tk  # Mantener referencia
    lbl_img.pack(side="left", padx=10, pady=10)

    # Panel derecho con controles
    frame = tk.Frame(ventana)
    frame.pack(side="right", padx=20, pady=20)

    etiqueta = tk.Label(frame, text="Seleccione cantidad de ruido:", font=("Arial", 12))
    etiqueta.pack(pady=10)

    slider_ruido = tk.Scale(frame, from_=0, to=100, orient="horizontal", length=250)
    slider_ruido.set(5)
    slider_ruido.pack()

    boton = ttk.Button(frame, text="Aplicar Filtro Mediana", command=aplicar_filtro)
    boton.pack(pady=20)

    ventana.mainloop()

# ---------------- EJECUTAR ----------------
iniciar_programa()
