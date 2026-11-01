import cv2
import numpy as np
import pydicom


def leer_imagen(ruta_archivo: str) -> np.ndarray:
    """
    Lee una imagen desde el disco. Soporta formatos JPEG/PNG y archivos médicos DICOM (.dcm).
    Responsabilidad única: Cargar los datos crudos en un arreglo de NumPy.
    """
    if ruta_archivo.lower().endswith('.dcm'):
        # Leer archivo médico DICOM
        ds = pydicom.dcmread(ruta_archivo)
        imagen = ds.pixel_array.astype(np.float32)
        # Asegurar que esté escalado correctamente a rango entero inicial si es necesario
        imagen = cv2.normalize(imagen, None, 0, 255, cv2.NORM_MINMAX)
        return imagen.astype(np.uint8)
    else:
        # Leer imagen estándar (JPEG/PNG) en escala de grises
        imagen = cv2.imread(ruta_archivo, cv2.IMREAD_GRAYSCALE)
        if imagen is None:
            raise ValueError(f"No se pudo leer la imagen o formato inválido en: '{ruta_archivo}'")
        return imagen
