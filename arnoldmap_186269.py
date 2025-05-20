from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import correlate2d

def arnold_cat_map(image, iterations):
    """Aplica el Arnold Cat Map a una imagen dada el número de iteraciones."""
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import correlate2d # Todavía lo importamos si quieres la correlación cruzada original
import scipy.stats # Necesario para pearsonr

def arnold_cat_map(image, iterations):
    """Aplica el Arnold Cat Map a una imagen dada el número de iteraciones."""
    n = image.shape[0]
    transformed_image = np.zeros_like(image)
    for _ in range(iterations):
        for x in range(n):
            for y in range(n):
                new_x = (x + y) % n
                new_y = (x + 2 * y) % n
                transformed_image[new_x, new_y] = image[x, y]
        image = transformed_image.copy()  # Actualizar la imagen para la siguiente iteración
    return transformed_image

def calculate_correlation_pearson(image1, image2):
    """
    Calcula el coeficiente de correlación de Pearson entre dos imágenes.
    Devuelve un valor entre -1 y 1.
    """
    # Aplanar las imágenes a vectores 1D
    flat_image1 = image1.flatten()
    flat_image2 = image2.flatten()

    # Calcular el coeficiente de correlación de Pearson
    # scipy.stats.pearsonr devuelve la correlación y el p-valor
    correlation, _ = scipy.stats.pearsonr(flat_image1, flat_image2)
    return correlation

if __name__ == "__main__":
    # Especifica la ruta a tu imagen
    ruta_imagen = "arnold.png"  # Asegúrate de que tu imagen "arnold.png" esté en la misma carpeta

    try:
        imagen_pil = Image.open(ruta_imagen).convert('L')  # Abre la imagen y la convierte a escala de grises
        original_image = np.array(imagen_pil)
        n = original_image.shape[0]

        if original_image.shape[0] != original_image.shape[1]:
            raise ValueError("La imagen debe ser cuadrada para el Arnold Cat Map.")

        if n > 255:
            print(f"Advertencia: La dimensión de la imagen ({n}) es mayor que 255. El código la redimensionará a 255x255.")
            imagen_pil = imagen_pil.resize((255, 255))
            original_image = np.array(imagen_pil)
            n = 255
            print("La imagen ha sido redimensionada a 255x255.")
        elif n < 255:
            print(f"Advertencia: La dimensión de la imagen ({n}) es menor que 255. Se recomienda usar imágenes de 255x255 para este ejemplo.")

    except FileNotFoundError:
        print(f"Error: No se encontró la imagen en la ruta: {ruta_imagen}")
        exit()
    except ValueError as e:
        print(f"Error: {e}")
        exit()
    except Exception as e:
        print(f"Ocurrió un error al cargar la imagen: {e}")
        exit()

    # 2. Aplicar el Arnold Cat Map varias veces
    iterations_to_mix = 50  # Un número de iteraciones para "mezclar" la imagen
    mixed_image = arnold_cat_map(original_image.copy(), iterations_to_mix)

    # 3. Encontrar el número de iteraciones para regresar a la imagen original
    recovery_iterations = 0
    recovered_image = mixed_image.copy()
    max_iterations_check = n * 10 # Limite para evitar bucles infinitos en casos raros
    
    while not np.array_equal(recovered_image, original_image):
        recovered_image = arnold_cat_map(recovered_image, 1)
        recovery_iterations += 1
        if recovery_iterations >= max_iterations_check:
            print(f"Advertencia: Se alcanzó el límite de {max_iterations_check} iteraciones para la recuperación. Puede que no se recupere la imagen original o el período es muy largo.")
            break

    print(f"Número de iteraciones para recuperar la imagen original: {recovery_iterations}")

    # 4. Calcular la correlación de Pearson entre la imagen original y la imagen mezclada
    correlation_mixed = calculate_correlation_pearson(original_image, mixed_image)
    print(f"Correlación de Pearson entre la imagen original y la imagen mezclada ({iterations_to_mix} iteraciones): {correlation_mixed:.4f}") # Formateado a 4 decimales

    # También podemos calcular la correlación de la imagen original con la imagen recuperada
    correlation_recovered = calculate_correlation_pearson(original_image, recovered_image)
    print(f"Correlación de Pearson entre la imagen original y la imagen recuperada: {correlation_recovered:.4f}")


    # 5. Visualizar las imágenes
    plt.figure(figsize=(15, 5))

    plt.subplot(1, 3, 1)
    plt.imshow(original_image, cmap='gray')
    plt.title("Imagen Original")

    plt.subplot(1, 3, 2)
    plt.imshow(mixed_image, cmap='gray')
    plt.title(f"Imagen Mezclada ({iterations_to_mix} iteraciones)\nCorrelación: {correlation_mixed:.4f}")

    plt.subplot(1, 3, 3)
    plt.imshow(recovered_image, cmap='gray')
    plt.title(f"Imagen Recuperada ({recovery_iterations} iteraciones)\nCorrelación: {correlation_recovered:.4f}")

    plt.tight_layout()
    plt.show()