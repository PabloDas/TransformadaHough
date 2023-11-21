from skimage.transform import (hough_line, hough_line_peaks)
import numpy as np
import cv2
from matplotlib import pyplot as plt

image = cv2.imread('images/motor.jpg', 0)  # Falla si se utiliza tal cual debido al fondo brillante.
# Prueba también lines2 para ver cómo sólo capta las líneas rectas
# Invertir imágenes para mostrar fondo negro
image = ~image  # Invertir la imagen (sólo si tenía fondo brillante que puede confundir hough)
plt.imshow(image, cmap='gray')

# Establezca una precisión de 1 grado. (Dividir en 180 puntos de datos)
# Puedes aumentar el número de puntos si es necesario.
tested_angles = np.linspace(-np.pi / 2, np.pi / 2, 10)

# Realiza la Transformación de Hough para cambiar x, y, a h, theta, dist space.
hspace, theta, dist = hough_line(image, tested_angles)

plt.figure(figsize=(10, 10))
plt.imshow(hspace)

# ahora, para encontrar la ubicación de los picos en el espacio de hough podemos utilizar hough_line_peaks
h, q, d = hough_line_peaks(hspace, theta, dist)

#################################################################
# Ejemplo de código de la documentación de skimage para trazar las líneas detectadas
angle_list = []  # Crear una lista vacía para capturar todos los ángulos

# Generación de la figura 1
fig, axes = plt.subplots(1, 3, figsize=(15, 6))
ax = axes.ravel()

ax[0].imshow(image, cmap='gray')
ax[0].set_title('Input image')
ax[0].set_axis_off()

ax[1].imshow(np.log(1 + hspace),
             extent=[np.rad2deg(theta[-1]), np.rad2deg(theta[0]), dist[-1], dist[0]],
             cmap='gray', aspect=1 / 1.5)
ax[1].set_title('Hough transform')
ax[1].set_xlabel('Angles (degrees)')
ax[1].set_ylabel('Distance (pixels)')
ax[1].axis('image')

ax[2].imshow(image, cmap='gray')

origin = np.array((0, image.shape[1]))

for _, angle, dist in zip(*hough_line_peaks(hspace, theta, dist)):
    angle_list.append(angle)
    # No para el trazado, sino para el cálculo posterior de ángulos
    y0, y1 = (dist - origin * np.cos(angle)) / np.sin(angle)
    ax[2].plot(origin, (y0, y1), '-r')
ax[2].set_xlim(origin)
ax[2].set_ylim((image.shape[0], 0))
ax[2].set_axis_off()
ax[2].set_title('Detected lines')

plt.tight_layout()
plt.show()

###############################################################
# Convertir ángulos de radianes a grados (1 rad = 180/pi grados)
angles = [a * 18.0 / np.pi for a in angle_list]

# Calcular la diferencia entre las dos líneas
angle_difference = np.max(angles) - np.min(angles)
print(180 - angle_difference)
# Restando de 180 para mostrarlo como el pequeño ángulo entre dos línea
