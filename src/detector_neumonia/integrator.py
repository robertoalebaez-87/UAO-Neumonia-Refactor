from collections.abc import Callable
from typing import Any

import numpy as np

from .preprocess_img import preprocesar_radiografia

# Importaciones relativas internas del proyecto
from .read_img import leer_imagen


class CoordinadorInferencia:
    """
    Controlador central (Orquestador).
    Conecta la lectura, el preprocesamiento, la predicción y la explicación.
    No interactúa con componentes gráficos de interfaz de usuario.
    """
    def __init__(self, modelo_ia: Any, funcion_grad_cam: Callable[..., np.ndarray]):
        self.modelo_ia = modelo_ia
        self.funcion_grad_cam = funcion_grad_cam

    def procesar_y_predecir(self, ruta_archivo: str) -> dict[str, Any]:
        """
        Orquesta el flujo secuencial completo de procesamiento e inferencia.
        """
        # 1. Leer los datos crudos desde el disco (JPEG, PNG o DICOM)
        img_cruda = leer_imagen(ruta_archivo)
        
        # 2. Transformar matemáticamente la imagen al formato requerido
        tensor_preprocesado = preprocesar_radiografia(img_cruda)
        
        # 3. Ejecutar la inferencia a través del modelo de IA cargado
        # El modelo retorna la probabilidad (ej: [[0.85]])
        prediccion_raw = self.modelo_ia.predict(tensor_preprocesado)
        probabilidad_neumonia = float(prediccion_raw[0][0])
        
        # Clasificación binaria con umbral estándar de 0.5
        etiqueta = "NEUMONÍA DETECTADA" if probabilidad_neumonia >= 0.5 else "NORMAL (SIN HALLAZGOS)"
        
        # 4. Generar la explicación visual mediante mapa de calor
        imagen_explicativa_rgb = self.funcion_grad_cam(self.modelo_ia, tensor_preprocesado)
        
        # Recuperar una versión visual 512x512 para mostrar en la interfaz como "imagen cargada"
        imagen_original_mostrable = np.uint8(tensor_preprocesado[0, :, :, 0] * 255)
        
        return {
            "probabilidad": probabilidad_neumonia,
            "resultado_texto": etiqueta,
            "imagen_original": imagen_original_mostrable,
            "imagen_explicada": imagen_explicativa_rgb
        }
