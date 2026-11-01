import cv2
import numpy as np
import tensorflow as tf


def generar_grad_cam(modelo: tf.keras.Model, tensor_imagen: np.ndarray, nombre_capa_convolucional: str | None = None) -> np.ndarray:
    """
    Calcula el mapa de activación por gradientes (Grad-CAM) para explicar la predicción.
    Responsabilidad única: Generar un mapa de calor superpuesto sobre la imagen original.
    """
    # Si no se especifica capa, buscar de forma automática la última capa convolucional 2D del modelo
    if nombre_capa_convolucional is None:
        for capa in reversed(modelo.layers):
            if isinstance(capa, tf.keras.layers.Conv2D) or "conv" in capa.name.lower():
                nombre_capa_convolucional = capa.name
                break
                
    if nombre_capa_convolucional is None:
        raise ValueError("No se pudo identificar una capa convolucional en el modelo.")

    # Construir un modelo secundario que mapee la entrada a las activaciones de la capa seleccionada y a la salida final
    modelo_grad = tf.keras.models.Model(
        inputs=[modelo.inputs],
        outputs=[modelo.get_layer(nombre_capa_convolucional).output, modelo.output]
    )

    # Grabar operaciones para calcular gradientes matemáticos
    with tf.GradientTape() as tape:
        activaciones_capa, predicciones = modelo_grad(tensor_imagen)
        # Tomar la salida de la clase predecida (neurona indexada en 0 habitualmente para clasificación binaria)
        score_clase = predicciones[:, 0]

    # Calcular gradientes del score de la clase con respecto a los mapas de características de la capa convolucional
    gradientes = tape.gradient(score_clase, activaciones_capa)

    # Reducir los gradientes a un vector de pesos mediante el promedio global en los ejes espaciales (H, W)
    pesos = tf.reduce_mean(gradientes, axis=(0, 1, 2))

    # Realizar una combinación lineal ponderada de las características de la capa usando los pesos calculados
    activaciones_capa = activaciones_capa[0]
    cam = np.zeros(activaciones_capa.shape[0:2], dtype=np.float32)

    for i, w in enumerate(pesos):
        cam += w * activaciones_capa[:, :, i]

    # Aplicar una función ReLU (fijar valores negativos en 0) y normalizar al rango [0, 255]
    cam = np.maximum(cam, 0)
    if np.max(cam) != 0:
        cam = cam / np.max(cam)
    cam = np.uint8(255 * cam)

    # Recuperar la imagen base desde el lote para poder fusionar el mapa de calor
    img_base = np.uint8(tensor_imagen[0, :, :, 0] * 255)
    
    # Redimensionar el mapa de calor de baja resolución al tamaño original de la imagen (512x512)
    cam_redimensionado = cv2.resize(cam, (img_base.shape[1], img_base.shape[0]))
    
    # Aplicar una paleta de colores JET (Azul a Rojo) sobre el mapa en escala de grises
    mapa_calor_color = cv2.applyColorMap(cam_redimensionado, cv2.COLORMAP_JET)

    # Convertir la imagen base a tres canales para poder sumar el color
    img_base_rgb = cv2.cvtColor(img_base, cv2.COLOR_GRAY2BGR)
    
    # Superponer ambas imágenes combinando el 60% de la original con el 40% del mapa de calor
    imagen_explicada = cv2.addWeighted(img_base_rgb, 0.6, mapa_calor_color, 0.4, 0)
    
    # Convertir a formato estándar RGB para la interfaz visual
    return cv2.cvtColor(imagen_explicada, cv2.COLOR_BGR2RGB)
