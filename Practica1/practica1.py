import cv2
import numpy as np
import matplotlib.pyplot as plt


imagen1 = cv2.imread("Practica1/ri.png", cv2.IMREAD_COLOR)
imagen2 = cv2.imread("Practica1/letra.png", cv2.IMREAD_COLOR)

alto, ancho = imagen1.shape[:2]

R = imagen1[:, :, 2]
G = imagen1[:, :, 1]
B = imagen1[:, :, 0]

resul_vert = np.zeros((alto, ancho, 3), dtype=np.uint8)
resul_hori = np.zeros((alto, ancho, 3), dtype=np.uint8)

tercio_hori = ancho // 3
tercio_vert = alto // 3

resul_vert[:, :tercio_hori, 2] = R[:, :tercio_hori]
resul_vert[:, tercio_hori : 2 * tercio_hori, 1] = G[:, tercio_hori : 2 * tercio_hori]
resul_vert[:, 2 * tercio_hori :, 0] = B[:, 2 * tercio_hori :]

resul_hori[:tercio_vert, :, 2] = R[:tercio_vert, :]
resul_hori[tercio_vert : 2 * tercio_vert, :, 1] = G[tercio_vert : 2 * tercio_vert, :]
resul_hori[2 * tercio_vert :, :, 0] = B[2 * tercio_vert :, :]

# Convertir a escala de grises si es necesario
if canales == 3:
    imagen_gray = cv2.cvtColor(imagen2, cv2.COLOR_BGR2GRAY)
else:
    imagen_gray = imagen2

# 1. Crear la máscara binaria (letra blanca sobre fondo negro)
BW = imagen_gray > 200

alto, ancho = BW.shape

# 2. Crear el fondo coloreado
fondo_coloreado = np.zeros((alto, ancho, 3), dtype=np.uint8)
ancho_rojo = int(ancho * 0.3)  # Ajusta el porcentaje si lo deseas

# Fondo rojo a la izquierda
fondo_coloreado[:, :ancho_rojo, 2] = 255  # Canal rojo (OpenCV usa BGR)

# Fondo azul a la derecha
fondo_coloreado[:, ancho_rojo:, 0] = 255  # Canal azul

# 3. Superponer la letra en verde
letra_verde = np.zeros((alto, ancho, 3), dtype=np.uint8)
letra_verde[:, :, 1] = BW.astype(np.uint8) * 255  # Canal verde

# 4. Combinar fondo y letra
resul_final = fondo_coloreado.copy()
resul_final[BW] = letra_verde[BW]

plt.figure(figsize=(12, 8))
plt.subplot(2, 2, 1)
plt.imshow(cv2.cvtColor(imagen1, cv2.COLOR_BGR2RGB))
plt.title("Imagen Original")
plt.axis("off")

plt.subplot(2, 2, 2)
plt.imshow(cv2.cvtColor(resul_hori, cv2.COLOR_BGR2RGB))
plt.title("Resultado Horizontal")
plt.axis("off")

plt.subplot(2, 2, 3)
plt.imshow(cv2.cvtColor(resul_vert, cv2.COLOR_BGR2RGB))
plt.title("Resultado Vertical")
plt.axis("off")

plt.subplot(2, 2, 4)
plt.imshow(cv2.cvtColor(resul_final, cv2.COLOR_BGR2RGB))
plt.title("Resultado Final")
plt.axis("off")

plt.tight_layout()
plt.show()
