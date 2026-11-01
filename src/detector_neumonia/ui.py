import tkinter as tk
from tkinter import filedialog, messagebox

from PIL import Image, ImageTk

from .grad_cam import generar_grad_cam
from .integrator import CoordinadorInferencia

# Importaciones de la arquitectura interna (Módulos previos)
from .load_model import cargar_modelo_neumonia


class VentanaPrincipal(tk.Tk):
    def __init__(self, coordinador: CoordinadorInferencia):
        super().__init__()
        self.coordinador = coordinador
        
        # Configuración básica de la ventana
        self.title("UAO Virtual - Sistema Integrado de Detección de Neumonía")
        self.geometry("900x600")
        self.configure(bg="#f5f6fa")
        
        self.crear_componentes()

    def crear_componentes(self):
        # Panel superior: Título e instrucciones
        panel_titulo = tk.Frame(self, bg="#2f3640", height=60)
        panel_titulo.pack(fill=tk.X, side=tk.TOP)
        
        label_titulo = tk.Label(
            panel_titulo, 
            text="DETECCIÓN AUTOMÁTICA DE NEUMONÍA EN RADIOGRAFÍAS", 
            fg="white", bg="#2f3640", font=("Helvetica", 14, "bold")
        )
        label_titulo.pack(pady=15)

        # Panel inferior: Botón para cargar imagen
        panel_controles = tk.Frame(self, bg="#dcdde1", height=50)
        panel_controles.pack(fill=tk.X, side=tk.BOTTOM)
        
        btn_cargar = tk.Button(
            panel_controles, text="Seleccionar Radiografía (JPEG/DICOM)", 
            command=self.cargar_y_procesar, bg="#44bd32", fg="white",
            font=("Helvetica", 11, "bold"), padx=10, pady=5
        )
        btn_cargar.pack(pady=10)

        # Panel central: Contenedor de visualizaciones y resultados
        self.panel_central = tk.Frame(self, bg="#f5f6fa")
        self.panel_central.pack(fill=tk.BOTH, expand=True, pady=10)

        # Sección izquierda: Imagen Original
        self.frame_izq = tk.LabelFrame(self.panel_central, text=" Radiografía Cargada ", font=("Helvetica", 10, "bold"))
        self.frame_izq.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=15, pady=5)
        self.lbl_img_original = tk.Label(self.frame_izq, text="Ninguna imagen cargada", fg="grey")
        self.lbl_img_original.pack(fill=tk.BOTH, expand=True)

        # Sección derecha: Imagen Explicada (Grad-CAM) y diagnóstico
        self.frame_der = tk.LabelFrame(self.panel_central, text=" Análisis de IA (Grad-CAM) ", font=("Helvetica", 10, "bold"))
        self.frame_der.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=15, pady=5)
        self.lbl_img_explicada = tk.Label(self.frame_der, text="Esperando procesamiento...", fg="grey")
        self.lbl_img_explicada.pack(fill=tk.BOTH, expand=True)

        # Barra de Estado inferior para texto del diagnóstico
        self.lbl_resultado = tk.Label(
            self, text="Estado: Listo para recibir radiografías.", 
            bd=1, relief=tk.SUNKEN, anchor=tk.W, font=("Helvetica", 12, "bold"), bg="#353b48", fg="white", pady=8
        )
        self.lbl_resultado.pack(fill=tk.X, side=tk.BOTTOM)

    def cargar_y_procesar(self):
        # Abrir el explorador de archivos nativo filtrando por extensiones válidas
        ruta_archivo = filedialog.askopenfilename(
            title="Seleccionar Radiografía",
            filetypes=[("Archivos Soportados", "*.jpg *.jpeg *.png *.dcm"), ("Imágenes Estándar", "*.jpg *.jpeg *.png"), ("Archivos Médicos DICOM", "*.dcm")]
        )
        
        if not ruta_archivo:
            return  # El usuario canceló la selección
        
        try:
            # Invocar al Controlador central para ejecutar todo el pipeline de análisis
            resultado = self.coordinador.procesar_y_predecir(ruta_archivo)
            
            # --- Actualizar Vista de la Imagen Original ---
            img_org_pil = Image.fromarray(resultado["imagen_original"]).resize((380, 380))
            img_org_tk = ImageTk.PhotoImage(image=img_org_pil)
            self.lbl_img_original.config(image=img_org_tk, text="")
            self.lbl_img_original.image = img_org_tk  # Mantener referencia en memoria

            # --- Actualizar Vista de la Imagen Explicativa (Grad-CAM) ---
            img_exp_pil = Image.fromarray(resultado["imagen_explicada"]).resize((380, 380))
            img_exp_tk = ImageTk.PhotoImage(image=img_exp_pil)
            self.lbl_img_explicada.config(image=img_exp_tk, text="")
            self.lbl_img_explicada.image = img_exp_tk  # Mantener referencia en memoria
            
            # --- Actualizar Etiquetas de Diagnóstico Fines Evaluativos ---
            porcentaje = resultado["probabilidad"] * 100
            texto_diagnostico = f" DIAGNÓSTICO: {resultado['resultado_texto']} | Confianza de la IA: {porcentaje:.2f}%"
            self.lbl_resultado.config(text=texto_diagnostico)
            
        except FileNotFoundError as fnf:
            messagebox.showerror("Error de Configuración", str(fnf))
        except Exception as e:  # noqa: BLE001
            messagebox.showerror("Error en Procesamiento", f"Ocurrió un error inesperado al analizar el archivo:\n{e!s}")

def ejecutar_aplicacion():
    """
    Punto de entrada principal. Carga el modelo, instancia el controlador e inicia la UI.
    """
    try:
        # 1. Cargar el modelo de IA aislado
        modelo = cargar_modelo_neumonia()
        # 2. Instanciar el Controlador (Orquestador) inyectando sus dependencias
        coordinador = CoordinadorInferencia(modelo_ia=modelo, funcion_grad_cam=generar_grad_cam)
        # 3. Lanzar el bucle principal de la UI
        app = VentanaPrincipal(coordinador)
        app.mainloop()
    except FileNotFoundError as e:
        # Manejo elegante si el estudiante no ha descargado los pesos en la raíz
        print(f"\n[ALERTA] {e!s}\nLa interfaz gráfica no iniciará hasta corregir esto.")
        # Crear una mini ventana de advertencia nativa
        root = tk.Tk()
        root.withdraw()
        messagebox.showerror("Archivo de Pesos Faltante", f"{e!s}\n\nDescarga el archivo e inténtalo de nuevo.")
        root.destroy()

if __name__ == "__main__":
    ejecutar_aplicacion()
