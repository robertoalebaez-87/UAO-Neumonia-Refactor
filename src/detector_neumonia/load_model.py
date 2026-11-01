import os

import tensorflow as tf


def cargar_modelo_neumonia(ruta_modelo: str = "conv_MLP_84.h5") -> tf.keras.Model:
    """
    Carga el modelo entrenado de TensorFlow/Keras desde la ruta especificada.
    Responsabilidad única: Proveer la instancia del modelo cargado.
    """
    if not os.path.exists(ruta_modelo):
        raise FileNotFoundError(
            f"No se encontró el archivo del modelo en: '{ruta_modelo}'. "
            f"Asegúrate de descargarlo e incluirlo en la raíz del proyecto."
        )
    
    # Cargar el modelo h5 de forma aislada
    modelo = tf.keras.models.load_model(ruta_modelo, compile=False)
    return modelo
