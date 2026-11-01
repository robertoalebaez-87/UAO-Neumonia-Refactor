import numpy as np

from src.detector_neumonia.preprocess_img import preprocesar_radiografia


def test_preprocesamiento_debe_retornar_dimensiones_correctas():
    """
    Prueba unitaria para verificar que el preprocesamiento adapte la imagen
    al formato exacto de entrada del modelo (1, 512, 512, 1).
    """
    # 1. Arrange (Preparar el escenario)
    imagen_simulada = np.random.randint(0, 256, (100, 100), dtype=np.uint8)

    # 2. Act (Ejecutar la acción a probar)
    tensor_salida = preprocesar_radiografia(imagen_simulada)

    # 3. Assert (Verificar que los resultados sean los esperados)
    assert tensor_salida.shape == (1, 512, 512, 1), f"Forma incorrecta: {tensor_salida.shape}"


def test_preprocesamiento_debe_normalizar_valores_entre_cero_y_uno():
    """
    Prueba unitaria para verificar que la intensidad de los píxeles
    quede normalizada matemáticamente en el rango estricto de [0.0, 1.0].
    """
    # 1. Arrange (Preparar el escenario)
    imagen_simulada = np.random.randint(0, 256, (200, 300), dtype=np.uint8)

    # 2. Act (Ejecutar la acción a probar)
    tensor_salida = preprocesar_radiografia(imagen_simulada)

    # 3. Assert (Verificar que el mínimo y máximo cumplan el rango)
    valor_minimo = np.min(tensor_salida)
    valor_maximo = np.max(tensor_salida)

    assert valor_minimo >= 0.0, f"El valor mínimo {valor_minimo} es menor que 0.0"
    assert valor_maximo <= 1.0, f"El valor máximo {valor_maximo} es mayor que 1.0"
