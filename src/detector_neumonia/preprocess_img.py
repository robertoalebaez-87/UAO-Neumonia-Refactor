import cv2
import numpy as np


def preprocesar_radiografia(imagen: np.ndarray) -> np.ndarray:
    """
    Ajusta las dimensiones, mejora el contraste y normaliza la imagen de entrada.
    Responsabilidad única: Retornar un tensor listo para inferencia (1, 512, 512, 1).
    No lee archivos del disco ni abre interfaces gráficas.
    """
    # 1. Asegurar escala de grises de un solo canal
    if len(imagen.shape) == 3:
        imagen = cv2.cvtColor(imagen, cv2.COLOR_BGR2GRAY)
        
    # 2. Redimensionar a la resolución requerida por el modelo (512x512)
    img_redimensionada = cv2.resize(imagen, (512, 512), interpolation=cv2.INTER_AREA)
    
    # 3. Aplicar ecualización adaptativa de histograma (CLAHE) para resaltar estructuras pulmonares
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    img_filtrada = clahe.apply(img_redimensionada)
    
    # 4. Normalizar la intensidad de píxeles al rango matemático [0.0, 1.0]
    img_normalizada = img_filtrada.astype(np.float32) / 255.0
    
    # 5. Expandir dimensiones para cumplir con el formato de lote (Batch, Height, Width, Channels) -> (1, 512, 512, 1)
    tensor_salida = np.expand_dims(img_normalizada, axis=(0, -1))
    
    return tensor_salida
