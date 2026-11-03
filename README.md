# INTEGRANTES
# Carolina Dorado - Carolina.dorado@uao.edu.co
# Roberto Baez   - Roberto_ale.baez@uao.edu.co
# Jorge Cifuentes - jorge_alb.cifuentes@uao.edu.co

# Sistema Integrado de Detección de Neumonía - UAO Virtual

Este proyecto ha sido refactorizado bajo un patrón arquitectónico **Modelo-Vista-Controlador (MVC)**, logrando alta cohesión y bajo acoplamiento a partir de un script monolítico original de 251 líneas. Cuenta con empaquetado reproducible moderno mediante `uv` y validación automática de pruebas unitarias.

## 🛠️ Requerimientos y Herramientas del Entorno
* **Python**: Versión >= 3.12 (Ejecutándose sobre Python 3.13)
* **Gestor de Entorno**: `uv` (Astral) para la congelación y sincronización determinista de dependencias (`uv.lock`)
* **Calidad de Código**: `ruff` para análisis estático y formateo
* **Pruebas Automatizadas**: `pytest` bajo patrón AAA (Arrange-Act-Assert)

## 📦 Arquitectura de Módulos (`src/detector_neumonia/`)
1. **`load_model.py`**: Aísla la carga en memoria del archivo binario de pesos `.h5`.
2. **`read_img.py`**: Soporta la lectura de imágenes estándar (JPEG/PNG) y archivos médicos digitales en formato **DICOM (`.dcm`)**.
3. **`preprocess_img.py`**: Redimensiona a 512x512, convierte a escala de grises, aplica ecualización adaptativa CLAHE y normaliza intensidades en el rango [0.0, 1.0].
4. **`grad_cam.py`**: Extrae la matriz de activaciones por gradiente y superpone el mapa de calor explicativo sobre la radiografía.
5. **`integrator.py`**: Controlador que orquesta secuencialmente el pipeline de datos mediante contratos desacoplados (DIP).
6. **`ui.py`**: Interfaz gráfica en Tkinter. Se comunica única y exclusivamente con el Controlador.

## 🚀 Instrucciones de Ejecución Rápida (Menos de 5 minutos)

### 1. Descarga del Modelo de Pesos
Asegúrese de colocar el archivo binario entrenado `conv_MLP_84.h5` (o el correspondiente provisto por la cátedra) en la **raíz principal** del proyecto. *(El archivo está explícitamente ignorado en `.gitignore` para evitar exceder los límites de subida de Git)*.

### 2. Automatización mediante comandos Estándar
Si cuenta con la utilidad `make`, ejecute:
* **Instalar Entorno**: `make install` (Ejecuta internamente `uv sync`)
* **Lanzar la Aplicación**: `make run` (Inicia el módulo ejecutable de la interfaz)
* **Validación de Calidad y Tests**: `make check` (Corre `ruff` y `pytest`)

Si su sistema operativo no dispone de `make`, ejecute directamente los comandos nativos de `uv`:
```bash
# Sincronizar el entorno virtual
uv sync

# Ejecutar las pruebas unitarias y el linter
uv run ruff check .
uv run pytest

# Iniciar la interfaz gráfica
uv run python -m src.detector_neumonia.ui
```
