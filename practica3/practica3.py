import cv2
import numpy as np
import tkinter as tk
from tkinter import filedialog, ttk
from PIL import Image, ImageTk
import matplotlib.pyplot as plt


def leer_imagen():
    """Lee una imagen y la convierte a escala de grises"""
    root = tk.Tk()
    root.withdraw()
    ruta = filedialog.askopenfilename(
        title="Seleccionar imagen",
        filetypes=[("Archivos de imagen", "*.png;*.jpg;*.jpeg;*.bmp;*.tif")],
    )
    root.destroy()

    if not ruta:
        print("No seleccionaste ninguna imagen. Programa finalizado.")
        return None

    # Leer imagen en color y convertir a escala de grises
    imagen = cv2.imread(ruta)
    if imagen is None:
        print("No se pudo cargar la imagen.")
        return None

    imagen_gray = cv2.cvtColor(imagen, cv2.COLOR_BGR2GRAY)
    print(
        f"Imagen cargada: {imagen_gray.shape} - Rango: [{imagen_gray.min()}, {imagen_gray.max()}]"
    )
    return imagen_gray


def mostrar_imagenes(
    original,
    aproximada,
    error,
    titulo_original="Original",
    titulo_aproximada="Aproximada",
    titulo_error="Error",
):
    aproximada_display = normalizar_para_despliegue(aproximada)
    error_display = normalizar_para_despliegue(error)

    plt.figure(figsize=(12, 6))

    plt.subplot(1, 3, 1)
    plt.imshow(original, cmap="gray", vmin=0, vmax=255)
    plt.title(f"{titulo_original}\n[{original.min()}-{original.max()}]")
    plt.axis("off")

    plt.subplot(1, 3, 2)
    plt.imshow(aproximada_display, cmap="gray", vmin=0, vmax=255)
    plt.title(f"{titulo_aproximada}\n[{aproximada.min():.1f}-{aproximada.max():.1f}]")
    plt.axis("off")

    plt.subplot(1, 3, 3)
    plt.imshow(error_display, cmap="gray", vmin=0, vmax=255)
    plt.title(f"{titulo_error}\n[{error.min():.1f}-{error.max():.1f}]")
    plt.axis("off")

    plt.tight_layout()
    plt.show()


def normalizar_para_despliegue(matriz):
    """Aplica +128 y ajusta al rango [0,255] para visualización"""
    matriz_display = matriz + 128
    matriz_display = np.clip(matriz_display, 0, 255)
    return matriz_display.astype(np.uint8)


# --- Funciones por implementar la compresión ---
def comprimir_imagen(matriz, bits):
    """Función principal de compresión"""

    # Crear matriz de aproximación [P] con el mismo patrón del apunte
    aproximada = calcular_valores_desconocidos(matriz)
    print(f"Rango de [P]: [{aproximada.min():.2f}, {aproximada.max():.2f}]")

    # Calcular matriz de error [E] = [O] - [P]
    error = calcular_matriz_error(matriz, aproximada)
    print(f"Rango de [E]: [{error.min():.2f}, {error.max():.2f}]")

    return aproximada, error


def calcular_valores_desconocidos(matriz):
    """Implementa el método de promedios - Patrón del apunte CORREGIDO"""
    aproximada = matriz.copy().astype(np.float64)
    height, width = matriz.shape

    # Aplicar el patrón exacto del apunte
    for i in range(1, height - 1):  # Evitar bordes
        for j in range(1, width - 1):
            # Solo procesar las posiciones específicas del patrón
            if (i % 2 == 1) and (j % 2 == 1) and (i + 1 < height) and (j + 1 < width):
                # Patrón exacto del apunte:
                # [i-1, j-1] [i-1, j] [i-1, j+1]
                # [i, j-1]   [b1]     [b2]
                # [i+1, j-1] [b3]     [b4]

                # b1 = promedio de todos los conocidos alrededor
                conocidos_b1 = [
                    matriz[i - 1, j - 1],
                    matriz[i - 1, j],
                    matriz[i - 1, j + 1],
                    matriz[i, j - 1],
                    matriz[i + 1, j - 1],
                ]
                b1 = np.mean(conocidos_b1)
                aproximada[i, j] = b1

                # b2 = promedio de (b1 + vecinos arriba/derecha)
                conocidos_b2 = [b1, matriz[i - 1, j], matriz[i - 1, j + 1]]
                b2 = np.mean(conocidos_b2)
                aproximada[i, j + 1] = b2

                # b3 = promedio de (b1 + b2 + vecinos izquierda/abajo)
                conocidos_b3 = [b1, b2, matriz[i, j - 1], matriz[i + 1, j - 1]]
                b3 = np.mean(conocidos_b3)
                aproximada[i + 1, j] = b3

                # b4 = promedio de (b1 + b2 + b3)
                b4 = np.mean([b1, b2, b3])
                aproximada[i + 1, j + 1] = b4

    return aproximada


def calcular_matriz_error(original, aproximada):
    """[O] - [P]"""
    return original.astype(np.float64) - aproximada


def cuantizar_errores(matriz_error, bits):
    """Calcula θ y genera MEQ - CORREGIDO"""
    min_error = np.min(matriz_error)
    max_error = np.max(matriz_error)

    print(f"Error min: {min_error:.2f}, max: {max_error:.2f}")

    # Calcular θ (salto) correctamente
    theta = (max_error - min_error) / (2**bits)
    theta = max(theta, 1e-6)  # Evitar theta=0
    print(f"Theta (salto): {theta:.4f}")

    # Crear matriz MEQ correctamente
    meq = np.floor((matriz_error - min_error) / theta).astype(np.int32)

    # Asegurar que no exceda el rango permitido
    meq = np.clip(meq, 0, (2**bits) - 1)

    return meq


def reconstruir_imagen(meq, aproximada, matriz_error, bits):
    min_error = np.min(matriz_error)
    max_error = np.max(matriz_error)
    theta = (max_error - min_error) / (2**bits)
    theta = max(theta, 1e-6)  # Evitar theta=0

    # Reconstruir errores correctamente: MEQ⁻¹ = min_error + MEQ * theta + theta/2
    # El theta/2 es para tomar el punto medio del intervalo
    errores_reconstruidos = min_error + meq.astype(np.float64) * theta + theta / 2

    # Reconstruir imagen final: [P] + MEQ⁻¹
    reconstruida = aproximada + errores_reconstruidos

    return reconstruida


def interfaz_usuario(imagen):
    """Interfaz principal para selección de bits"""

    # Aplicar compresión
    imagen_aproximada, imagen_error = comprimir_imagen(imagen, 8)

    # Mostrar resultados
    mostrar_imagenes(
        imagen,
        imagen_aproximada,
        imagen_error,
        f"Original ({imagen.shape})",
        f"Aproximada ({imagen_aproximada.shape})",
        f"Error ({imagen_error.shape})",
    )

    def aplicar_compression():
        try:
            bits = int(entry_bits.get())
            if bits < 1 or bits > 8:
                raise ValueError("Bits fuera de rango")

            # Cuantizar errores para obtener MEQ
            meq = cuantizar_errores(imagen_error, bits)
            print(f"Rango de MEQ: [{meq.min()}, {meq.max()}] - {2**bits} niveles")

            reconstruida = reconstruir_imagen(
                meq, imagen_aproximada, imagen_error, bits
            )

            """Muestra las imágenes lado a lado para comparación"""
            # Calcular métricas de calidad
            mse = np.mean((imagen - reconstruida) ** 2)
            psnr = 10 * np.log10((255**2) / mse) if mse > 0 else float("inf")

            print(
                f"\nMSE (Error Cuadrático Medio): (1/n) * Σ(Originalᵢ - Comprimidaᵢ)² = {mse:.2f}"
            )
            print(
                f"PSNR (Relación Señal-Ruido Pico): 10 * log₁₀(MAX² / MSE) = {psnr:.2f} dB"
            )

            # Normalizar para visualización (escala 0-255) - sin sumar 128
            comprimida_display = np.clip(reconstruida, 0, 255).astype(np.uint8)

            # Mostrar solo la imagen comprimida
            plt.figure(figsize=(6, 6))
            plt.imshow(comprimida_display, cmap="gray", vmin=0, vmax=255)
            plt.title(
                f"Imagen comprimida\n[{comprimida_display.min()}-{comprimida_display.max()}]"
            )
            plt.axis("off")
            plt.tight_layout()
            plt.show()

        except ValueError as e:
            lbl_resultado.config(text=f"Error: {str(e)}", fg="red")
        except Exception as e:
            lbl_resultado.config(text=f"Error inesperado: {str(e)}", fg="red")

    # Crear ventana principal
    ventana = tk.Tk()
    ventana.title("Compresor de Imágenes")
    ventana.geometry("400x250")
    ventana.configure(bg="#f0f0f0")

    # Marco principal
    frame = tk.Frame(ventana, bg="#f0f0f0", padx=20, pady=20)
    frame.pack(expand=True, fill="both")

    # Título
    lbl_titulo = tk.Label(
        frame, text="Compresión de Imágenes", font=("Arial", 16, "bold"), bg="#f0f0f0"
    )
    lbl_titulo.pack(pady=(0, 20))

    # Información de la imagen
    info_text = f"Imagen: {imagen.shape[1]}x{imagen.shape[0]} píxeles"
    lbl_info = tk.Label(frame, text=info_text, font=("Arial", 10), bg="#f0f0f0")
    lbl_info.pack(pady=(0, 15))

    # Entrada para bits
    frame_bits = tk.Frame(frame, bg="#f0f0f0")
    frame_bits.pack(pady=10)

    lbl_bits = tk.Label(
        frame_bits, text="Bits por píxel (1-8):", font=("Arial", 11), bg="#f0f0f0"
    )
    lbl_bits.pack(side="left", padx=(0, 10))

    entry_bits = tk.Entry(frame_bits, font=("Arial", 11), width=5, justify="center")
    entry_bits.insert(0, "4")  # Valor por defecto
    entry_bits.pack(side="left")

    # Botón de compresión
    btn_comprimir = ttk.Button(
        frame,
        text="Comprimir Imagen",
        command=aplicar_compression,
        style="Accent.TButton",
    )
    btn_comprimir.pack(pady=15)

    # Etiqueta para resultados
    lbl_resultado = tk.Label(
        frame, text="", font=("Arial", 10), bg="#f0f0f0", fg="green"
    )
    lbl_resultado.pack(pady=5)

    # Configurar estilo moderno
    style = ttk.Style()
    style.configure("Accent.TButton", font=("Arial", 11, "bold"), padding=(20, 10))

    # Centrar ventana en la pantalla
    ventana.update_idletasks()
    width = ventana.winfo_width()
    height = ventana.winfo_height()
    x = (ventana.winfo_screenwidth() // 2) - (width // 2)
    y = (ventana.winfo_screenheight() // 2) - (height // 2)
    ventana.geometry(f"{width}x{height}+{x}+{y}")

    ventana.mainloop()


# Función principal temporal para testing
def main():
    print("=== COMPRESOR DE IMÁGENES ===")
    print("Presiona Ctrl+C en cualquier momento para salir del programa\n")

    while True:
        try:
            print("\n" + "=" * 50)
            print("Selecciona una nueva imagen para comprimir...")

            # Cargar imagen
            imagen = leer_imagen()
            if imagen is None:
                # Si el usuario cancela la selección, preguntar si quiere continuar
                continuar = (
                    input("\n¿Deseas seleccionar otra imagen? (s/n): ").lower().strip()
                )
                if continuar not in ["s", "si", "sí", "y", "yes"]:
                    break
                continue

            # Mostrar imagen original
            plt.imshow(imagen, cmap="gray")
            plt.title("Imagen Original")
            plt.axis("off")
            plt.show()

            # Mostrar interfaz de usuario
            interfaz_usuario(imagen)

            # Preguntar si quiere procesar otra imagen
            print("\n" + "=" * 50)
            continuar = input("¿Deseas procesar otra imagen? (s/n): ").lower().strip()
            if continuar not in ["s", "si", "sí", "y", "yes"]:
                break

        except KeyboardInterrupt:
            print("\n\nPrograma interrumpido por el usuario.")
            break
        except Exception as e:
            print(f"\nError inesperado: {e}")
            continuar = (
                input("¿Deseas continuar con otra imagen? (s/n): ").lower().strip()
            )
            if continuar not in ["s", "si", "sí", "y", "yes"]:
                break

    print("\n¡Gracias por usar el compresor de imágenes!")


if __name__ == "__main__":
    main()
