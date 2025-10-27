# Sistema de Captura de Movimiento del Brazo con OpenPose

## 📋 Descripción

Esta versión del sistema de captura de movimiento utiliza **OpenPose** de CMU (Carnegie Mellon University) para la detección de pose humana. OpenPose es un sistema de detección de pose en tiempo real que ofrece alta precisión y robustez en condiciones variadas.

## 🎯 Características

- **Detección basada en OpenPose**: Mayor precisión en la detección de landmarks corporales
- **Rango completo de movimiento**: -90° a 180° para elevación del hombro
- **Cálculo de ángulos en tiempo real**: Hombro y codo
- **Visualización mejorada**: Esqueleto completo con etiquetas
- **Compatible con OpenSim**: Exportación en formato `.mot`
- **Calibración personalizada**: Adaptable a cada usuario

## 📦 Requisitos

### Software
- Python 3.8 o superior
- Cámara web
- Sistema operativo: Windows, macOS, o Linux

### Dependencias Python
```bash
pip install opencv-python numpy
```

### Modelos de OpenPose (Requeridos)
Debes descargar 2 archivos (~200 MB en total):
1. `pose_deploy_linevec.prototxt` (~50 KB)
2. `pose_iter_440000.caffemodel` (~200 MB)

## 🚀 Instalación

### Paso 1: Clonar o Descargar el Proyecto
```bash
git clone <tu-repositorio>
cd opensim
```

### Paso 2: Instalar Dependencias
```bash
pip install -r requirements.txt
```

### Paso 3: Descargar Modelos de OpenPose

#### Opción A - Descarga Manual (Recomendado)

**Archivo 1: pose_deploy_linevec.prototxt**
1. Visita: https://raw.githubusercontent.com/CMU-Perceptual-Computing-Lab/openpose/master/models/pose/coco/pose_deploy_linevec.prototxt
2. Haz clic derecho → "Guardar como"
3. Guarda en: `models/pose/coco/pose_deploy_linevec.prototxt`

**Archivo 2: pose_iter_440000.caffemodel**
1. **Google Drive** (más fácil): https://drive.google.com/file/d/1XISgkmF6kpNCfQ4vfRj-qLwpzKmjLgWf/view
2. Descarga el archivo (~200 MB)
3. Guarda en: `models/pose/coco/pose_iter_440000.caffemodel`

#### Opción B - URLs Alternativas

**Archivo 1 (Alternativa):**
- https://github.com/opencv/opencv_extra/blob/master/testdata/dnn/openpose_pose_coco.prototxt

**Archivo 2 (Alternativas):**
- Sitio oficial: http://posefs1.perception.cs.cmu.edu/OpenPose/models/pose/coco/pose_iter_440000.caffemodel
- Dropbox: https://www.dropbox.com/s/2dw1oz9t6hkx7g8/pose_iter_440000.caffemodel?dl=1

### Paso 4: Verificar Estructura de Directorios

Tu proyecto debe verse así:
```
opensim/
├── models/
│   └── pose/
│       └── coco/
│           ├── pose_deploy_linevec.prototxt  (~50 KB)
│           └── pose_iter_440000.caffemodel   (~200 MB)
├── openpose.py
├── requirements.txt
└── README_OPENPOSE.md
```

## 🎮 Uso del Sistema

### Ejecutar el Programa
```bash
python openpose.py
```

### Controles
| Tecla | Función |
|-------|---------|
| `c` | Calibrar posición inicial del brazo |
| `r` | Iniciar/Detener grabación |
| `s` | Guardar datos en archivo .mot |
| `q` | Salir del programa |

### Flujo de Trabajo
1. **Iniciar**: Ejecuta `python openpose.py`
2. **Posicionar**: Coloca tu brazo derecho frente a la cámara
3. **Calibrar**: Presiona `c` para establecer la posición de referencia
4. **Grabar**: Presiona `r` para iniciar la grabación
5. **Mover**: Realiza los movimientos del brazo
6. **Detener**: Presiona `r` nuevamente para detener
7. **Guardar**: Presiona `s` para exportar el archivo .mot
8. **Salir**: Presiona `q` para cerrar

## 📊 Arquitectura del Código

### Clase Principal: `ArmMotionCaptureOpenPose`

```python
class ArmMotionCaptureOpenPose:
    def __init__(self, openpose_model_path)
    def detect_pose(self, frame)
    def calculate_arm_angles(self, points)
    def draw_skeleton(self, frame, points)
    def calibrate_initial_position(self, points)
    def start_recording(self)
    def stop_recording(self)
    def save_motion_file(self, filename)
    def process_frame(self, frame, points)
    def run(self)
```

### Componentes Clave

#### 1. Inicialización del Modelo (`__init__`)
```python
self.net = cv2.dnn.readNetFromCaffe(self.proto_file, self.weights_file)
```
- Carga los modelos de OpenPose usando OpenCV DNN
- Configura parámetros de detección (umbral, dimensiones)
- Inicializa variables de control y grabación

#### 2. Detección de Pose (`detect_pose`)
```python
def detect_pose(self, frame):
    # Preparar entrada (368x368)
    inpBlob = cv2.dnn.blobFromImage(frame, 1.0/255, (368, 368))
    
    # Inferencia con la red neuronal
    self.net.setInput(inpBlob)
    output = self.net.forward()
    
    # Extraer puntos clave (hombro, codo, muñeca)
    # ...
    return points
```

**Puntos Detectados (Modelo BODY_25):**
- Índice 2: Hombro derecho (`RShoulder`)
- Índice 3: Codo derecho (`RElbow`)
- Índice 4: Muñeca derecha (`RWrist`)

#### 3. Cálculo de Ángulos (`calculate_arm_angles`)

**Elevación del Hombro (r_shoulder_elev):**
```python
# Vector del brazo (hombro → codo)
arm_vector_x = elbow_x - shoulder_x
arm_vector_y = elbow_y - shoulder_y

# Ángulo respecto a la vertical
shoulder_elev_raw = np.arctan2(arm_vector_x, arm_vector_y) * 180 / np.pi

# Invertir signo (subir = positivo)
shoulder_elev = -shoulder_elev_raw

# Limitar a rango -90° a 180°
shoulder_elev = np.clip(shoulder_elev, -90, 180)
```

**Rango:**
- **-90°**: Brazo completamente hacia abajo
- **0°**: Brazo horizontal (lateral)
- **90°**: Brazo hacia arriba (vertical)
- **180°**: Brazo hacia el otro lado (horizontal)

**Flexión del Codo (r_elbow_flex):**
```python
# Vectores: brazo superior y antebrazo
upper_arm = (elbow - shoulder)
forearm = (wrist - elbow)

# Ángulo entre vectores (producto punto)
cos_angle = dot_product / (length1 * length2)
elbow_flex = arccos(cos_angle) * 180 / π

# Compensar offset mínimo
elbow_flex = max(0, elbow_flex - 20)
```

**Rango:**
- **0°**: Brazo completamente extendido
- **90°**: Codo en ángulo recto
- **180°**: Brazo completamente flexionado

#### 4. Visualización (`draw_skeleton`)
```python
def draw_skeleton(self, frame, points):
    # Dibujar círculos en articulaciones
    cv2.circle(frame, shoulder_pos, 8, color, -1)
    cv2.circle(frame, elbow_pos, 8, color, -1)
    cv2.circle(frame, wrist_pos, 8, color, -1)
    
    # Dibujar líneas conectando articulaciones
    cv2.line(frame, shoulder_pos, elbow_pos, color, 3)
    cv2.line(frame, elbow_pos, wrist_pos, color, 3)
    
    # Etiquetas de texto
    cv2.putText(frame, "Hombro", shoulder_pos, ...)
```

#### 5. Exportación OpenSim (`save_motion_file`)

**Formato de Archivo .mot:**
```
Coordinates
version=1
nRows=150
nColumns=3
inDegrees=yes

Units are S.I. units (second, meters, Newtons, ...)
Angles are in degrees.

endheader
time    r_shoulder_elev    r_elbow_flex
0.00000000    15.23456789    25.67890123
0.03333333    15.45678901    25.89012345
...
```

## 🔧 Configuración Avanzada

### Ajustar Umbral de Confianza
```python
self.threshold = 0.1  # Valor por defecto
# Aumentar para mayor precisión (menos falsos positivos)
# Disminuir para mayor sensibilidad (más detecciones)
```

### Cambiar Dimensiones de Entrada
```python
self.inWidth = 368   # Por defecto
self.inHeight = 368  # Por defecto

# Opciones: 256x256 (más rápido), 368x368 (balanceado), 432x368 (más preciso)
```

### Configurar Cámara
```python
# Resolución de captura
self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

# Usar cámara específica (0, 1, 2, ...)
self.cap = cv2.VideoCapture(0)  # 0 = cámara por defecto
```

## 📈 Ventajas de OpenPose vs MediaPipe

| Característica | OpenPose | MediaPipe |
|----------------|----------|-----------|
| **Precisión** | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐ |
| **Velocidad** | ⭐⭐⭐ | ⭐⭐⭐⭐⭐ |
| **Robustez** | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐ |
| **Configuración** | ⭐⭐ | ⭐⭐⭐⭐⭐ |
| **Tamaño** | ~200 MB | ~100 MB |
| **Uso Académico** | ✅ Estándar | ✅ Popular |

## 🐛 Solución de Problemas

### Error: "No se puede acceder a la cámara"
**Causa:** Cámara no disponible o en uso
**Solución:**
```bash
# Verificar cámaras disponibles
ls /dev/video*  # Linux
# O cambiar índice: cv2.VideoCapture(1)
```

### Error: "Error cargando modelo OpenPose"
**Causa:** Archivos de modelo corruptos o faltantes
**Solución:**
1. Verifica que los archivos existan:
   ```bash
   ls -lh models/pose/coco/
   ```
2. Verifica el tamaño:
   - `pose_deploy_linevec.prototxt`: ~50 KB
   - `pose_iter_440000.caffemodel`: ~200 MB
3. Si están corruptos, elimina y descarga nuevamente

### Error: "OpenCV DNN: getMemoryShapes() throws exception"
**Causa:** Archivo `.caffemodel` corrupto o incompleto
**Solución:**
```bash
# Eliminar archivo corrupto
rm models/pose/coco/pose_iter_440000.caffemodel

# Descargar nuevamente desde Google Drive
# URL: https://drive.google.com/file/d/1XISgkmF6kpNCfQ4vfRj-qLwpzKmjLgWf/view
```

### Error: "No se detecta el brazo"
**Causa:** Iluminación pobre o pose no clara
**Solución:**
- Mejora la iluminación
- Asegúrate de que el brazo esté completamente visible
- Aleja la cámara para capturar más contexto
- Reduce el umbral: `self.threshold = 0.05`

### Detección Lenta
**Causa:** Hardware limitado
**Solución:**
```python
# Reducir resolución de entrada
self.inWidth = 256
self.inHeight = 256

# Usar GPU si está disponible
self.net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
self.net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)
```

## 📚 Recursos Adicionales

### Documentación Oficial
- **OpenPose GitHub**: https://github.com/CMU-Perceptual-Computing-Lab/openpose
- **OpenPose Paper**: https://arxiv.org/abs/1812.08008
- **OpenCV DNN Module**: https://docs.opencv.org/4.x/d2/d58/tutorial_table_of_content_dnn.html

### Modelos OpenPose
- **BODY_25**: 25 puntos corporales (usado en este proyecto)
- **COCO**: 18 puntos corporales
- **MPI**: 15 puntos corporales

### OpenSim
- **Documentación**: https://simtk-confluence.stanford.edu/display/OpenSim/Documentation
- **Modelo arm26**: https://simtk.org/projects/opensim

## 🤝 Comparación con main.py (MediaPipe)

### Cuándo Usar OpenPose (`openpose.py`)
✅ Necesitas máxima precisión
✅ Investigación académica
✅ Condiciones de iluminación variables
✅ Múltiples sujetos en escena
✅ Hardware potente disponible

### Cuándo Usar MediaPipe (`main.py`)
✅ Configuración rápida sin descargas
✅ Hardware limitado (laptops, dispositivos móviles)
✅ Necesitas velocidad en tiempo real
✅ Aplicaciones comerciales
✅ Primera vez usando el sistema

## 📝 Archivos Generados

### Estructura de Salida
```
motion_files/
├── arm26_motion_openpose_20241027_143022.mot
├── arm26_motion_openpose_20241027_145830.mot
└── final_motion_openpose.mot
```

### Formato de Archivo .mot
```
Coordinates
version=1
nRows=300
nColumns=3
inDegrees=yes

endheader
time	r_shoulder_elev	r_elbow_flex
0.00000000	15.34567890	28.12345678
0.03333333	15.67890123	28.45678901
...
```

## 🔬 Validación y Precisión

### Rangos Validados
- **Elevación del hombro**: -90° a 180°
- **Flexión del codo**: 0° a 180°
- **Frecuencia de muestreo**: ~30 Hz (dependiente de hardware)
- **Precisión estimada**: ±2-5° (dependiente de condiciones)

### Calibración
La calibración establece la posición de referencia específica del usuario:
- Compensa diferencias en postura natural
- Mejora consistencia de mediciones
- Requerida antes de cada sesión de grabación

## 📄 Licencia

Este proyecto utiliza OpenPose, que está bajo licencia específica de CMU.
Consulta: https://github.com/CMU-Perceptual-Computing-Lab/openpose/blob/master/LICENSE

## 👥 Contribuciones

Para contribuir al proyecto:
1. Fork del repositorio
2. Crea una rama: `git checkout -b feature/nueva-funcionalidad`
3. Commit cambios: `git commit -m 'Agregar nueva funcionalidad'`
4. Push a la rama: `git push origin feature/nueva-funcionalidad`
5. Abre un Pull Request

## 📞 Soporte

- **Issues**: [GitHub Issues]
- **Documentación**: Ver `DESCARGA_MANUAL.md` para problemas de instalación
- **Alternativa**: Usa `main.py` (MediaPipe) si tienes problemas con OpenPose

---

**Nota Importante**: Este sistema requiere descargar ~200 MB de modelos de OpenPose. Si prefieres una solución sin descargas adicionales, usa `main.py` que utiliza MediaPipe.
