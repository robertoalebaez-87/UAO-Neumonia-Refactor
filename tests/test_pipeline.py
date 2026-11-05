import numpy as np
import pytest
from unittest.mock import MagicMock, patch

from src.detector_neumonia.integrator import CoordinadorInferencia
from src.detector_neumonia.read_img import leer_imagen
from src.detector_neumonia.load_model import cargar_modelo_neumonia


# --- 1. Pruebas para la lectura de imágenes (read_img.py) ---

def test_leer_imagen_lanza_error_con_ruta_invalida():
    """Valida que lance ValueError si cv2 no puede leer la imagen."""
    with patch("cv2.imread", return_value=None):
        with pytest.raises(ValueError, match="No se pudo leer la imagen"):
            leer_imagen("ruta_inexistente.jpg")


def test_leer_imagen_dicom_mock():
    """Valida la lectura y normalización uint8 de un archivo médico DICOM simulado."""
    fake_dataset = MagicMock()
    fake_dataset.pixel_array = np.array([[10, 20], [30, 40]], dtype=np.int16)

    with patch("pydicom.dcmread", return_value=fake_dataset):
        resultado = leer_imagen("estudio_pulmon.dcm")
        assert isinstance(resultado, np.ndarray)
        assert resultado.dtype == np.uint8


# --- 2. Pruebas para la carga del modelo (load_model.py) ---

def test_cargar_modelo_lanza_filenotfound_si_no_existe():
    """Valida que informe claramente con FileNotFoundError si no existe el archivo .h5."""
    with patch("os.path.exists", return_value=False):
        with pytest.raises(FileNotFoundError, match="No se encontró el archivo del modelo"):
            cargar_modelo_neumonia("modelo_no_existente.h5")


# --- 3. Pruebas para el orquestador (integrator.py) ---

def test_coordinador_clasificacion_positiva():
    """Valida detección de neumonía cuando la probabilidad es >= 0.5."""
    mock_modelo = MagicMock()
    mock_modelo.predict.return_value = np.array([[0.85]])
    mock_grad_cam = MagicMock(return_value=np.zeros((512, 512, 3), dtype=np.uint8))
    dummy_img = np.zeros((100, 100), dtype=np.uint8)

    coordinador = CoordinadorInferencia(modelo_ia=mock_modelo, funcion_grad_cam=mock_grad_cam)

    with patch("src.detector_neumonia.integrator.leer_imagen", return_value=dummy_img):
        resultado = coordinador.procesar_y_predecir("radiografia_neumonia.jpg")

    assert resultado["resultado_texto"] == "NEUMONÍA DETECTADA"
    assert resultado["probabilidad"] == pytest.approx(0.85)
    assert "imagen_original" in resultado
    assert "imagen_explicada" in resultado


def test_coordinador_clasificacion_negativa():
    """Valida caso normal (sin hallazgos) cuando la probabilidad es < 0.5."""
    mock_modelo = MagicMock()
    mock_modelo.predict.return_value = np.array([[0.15]])
    mock_grad_cam = MagicMock(return_value=np.zeros((512, 512, 3), dtype=np.uint8))
    dummy_img = np.zeros((100, 100), dtype=np.uint8)

    coordinador = CoordinadorInferencia(modelo_ia=mock_modelo, funcion_grad_cam=mock_grad_cam)

    with patch("src.detector_neumonia.integrator.leer_imagen", return_value=dummy_img):
        resultado = coordinador.procesar_y_predecir("radiografia_sana.png")

    assert resultado["resultado_texto"] == "NORMAL (SIN HALLAZGOS)"
    assert resultado["probabilidad"] == pytest.approx(0.15)