# Sistema de Captura de Movimiento del Brazo para OpenSim

## 📋 Descripción del Proyecto

Este proyecto implementa un sistema de captura de movimiento del brazo humano utilizando visión por computadora y detección de pose. El sistema está diseñado para ser compatible con OpenSim, permitiendo la grabación y exportación de datos de movimiento en formato `.mot` para análisis biomecánico.

## 🎯 Características Principales

- **Detección en tiempo real** del movimiento del brazo derecho
- **Cálculo de ángulos** de elevación del hombro y flexión del codo
- **Calibración automática** de la posición inicial
- **Grabación de datos** con timestamps precisos
- **Exportación compatible** con OpenSim (formato `.mot`)
- **Interfaz visual** con feedback en tiempo real

## 🛠️ Tecnologías Utilizadas

- **OpenCV**: Captura y procesamiento de video
- **MediaPipe**: Detección de pose y landmarks corporales
- **NumPy**: Cálculos matemáticos y operaciones con arrays
- **Python 3.8+**: Lenguaje de programación principal

## 📦 Instalación

### Requisitos Previos
- Python 3.8 o superior
- Cámara web funcional
- Sistema operativo: Windows, macOS, o Linux

### Instalación de Dependencias
```bash
pip install -r requirements.txt
```

### Ejecución del Programa
```bash
python main.py
```

## 🎮 Controles del Sistema

| Tecla | Función |
|-------|---------|
| `c` | Calibrar posición inicial del brazo |
| `r` | Iniciar/detener grabación de movimiento |
| `s` | Guardar datos actuales en archivo .mot |
| `q` | Salir del programa |

## 📊 Estructura del Código

### 1. Clase Principal: `ArmMotionCapture`

La clase principal que maneja toda la funcionalidad del sistema.

```python
class ArmMotionCapture:
    def __init__(self):
        # Inicialización de MediaPipe y configuración
```

#### Componentes Principales:

**Inicialización de MediaPipe:**
```python
self.mp_pose = mp.solutions.pose
self.mp_drawing = mp.solutions.drawing_utils
self.pose = self.mp_pose.Pose(
    static_image_mode=False,
    model_complexity=1,
    enable_segmentation=False,
    min_detection_confidence=0.8,
    min_tracking_confidence=0.6
)
```

**Configuración de Articulaciones OpenSim:**
```python
self.opensim_joints = [
    'r_shoulder_elev',  # Elevación del hombro derecho
    'r_elbow_flex'      # Flexión del codo derecho
]
```

**Mapeo de Landmarks:**
```python
self.landmark_mapping = {
    'shoulder': [11, 12],  # Hombros izquierdo y derecho
    'elbow': [13, 14],     # Codos izquierdo y derecho
    'wrist': [15, 16],     # Muñecas izquierda y derecha
    'hand': [17, 18, 19, 20]  # Puntos de la mano
}
```

### 2. Cálculo de Ángulos: `calculate_arm_angles()`

Esta función es el corazón del sistema, calculando los ángulos del brazo en tiempo real.

#### Proceso de Cálculo:

**1. Obtención de Puntos Clave:**
```python
shoulder = landmarks[self.landmark_mapping['shoulder'][1]]  # Hombro derecho
elbow = landmarks[self.landmark_mapping['elbow'][1]]        # Codo derecho
wrist = landmarks[self.landmark_mapping['wrist'][1]]        # Muñeca derecha
```

**2. Conversión de Coordenadas:**
```python
shoulder_x, shoulder_y = shoulder.x, shoulder.y
elbow_x, elbow_y = elbow.x, elbow.y
wrist_x, wrist_y = wrist.x, wrist.y
```

**3. Cálculo de Elevación del Hombro (`r_shoulder_elev`):**
```python
# Vector del brazo (hombro a codo)
arm_vector_x = elbow_x - shoulder_x
arm_vector_y = elbow_y - shoulder_y

# Calcular ángulo respecto a la vertical
shoulder_elev_raw = np.arctan2(arm_vector_x, arm_vector_y) * 180 / np.pi

# Invertir el signo para que subir = ángulo positivo mayor
shoulder_elev = -shoulder_elev_raw
```

**Rango de Elevación del Hombro:**
- **-90°**: Brazo completamente hacia abajo (vertical)
- **0°**: Brazo horizontal hacia la derecha
- **90°**: Brazo hacia arriba (vertical)
- **180°**: Brazo horizontal hacia la izquierda

**4. Cálculo de Flexión del Codo (`r_elbow_flex`):**
```python
# Vector del brazo (hombro a codo)
upper_arm_x = elbow_x - shoulder_x
upper_arm_y = elbow_y - shoulder_y

# Vector del antebrazo (codo a muñeca)
forearm_x = wrist_x - elbow_x
forearm_y = wrist_y - elbow_y

# Calcular ángulo entre vectores usando producto punto
dot_product = upper_arm_x * forearm_x + upper_arm_y * forearm_y
upper_arm_length = np.sqrt(upper_arm_x**2 + upper_arm_y**2)
forearm_length = np.sqrt(forearm_x**2 + forearm_y**2)

cos_angle = dot_product / (upper_arm_length * forearm_length)
elbow_flex = np.arccos(cos_angle) * 180 / np.pi
```

**Rango de Flexión del Codo:**
- **0°**: Brazo completamente extendido
- **90°**: Codo flexionado a 90 grados
- **180°**: Brazo completamente flexionado

### 3. Sistema de Calibración: `calibrate_initial_position()`

```python
def calibrate_initial_position(self, landmarks):
    """Calibra la posición inicial del brazo"""
    try:
        angles = self.calculate_arm_angles(landmarks)
        if angles:
            self.calibrated = True
            print("✅ Calibración completada")
            print(f"   Elevación del hombro: {angles['r_shoulder_elev']:.1f}°")
            print(f"   Flexión del codo: {angles['r_elbow_flex']:.1f}°")
            return True
    except Exception as e:
        print(f"Error en calibración: {e}")
    return False
```

### 4. Sistema de Grabación

**Iniciar Grabación:**
```python
def start_recording(self):
    """Inicia la grabación de movimiento"""
    if not self.recording:
        self.recording = True
        self.motion_data = []
        self.start_time = time.time()
        print("🔴 Grabación iniciada")
```

**Detener Grabación:**
```python
def stop_recording(self):
    """Detiene la grabación de movimiento"""
    if self.recording:
        self.recording = False
        print("⏹️ Grabación detenida")
```

### 5. Exportación de Datos: `save_motion_file()`

**Formato OpenSim Compatible:**
```python
def save_motion_file(self, filename=None):
    """Guarda los datos de movimiento en formato .mot compatible con OpenSim arm26"""
    if not self.motion_data:
        print("No hay datos para guardar")
        return
    
    # Crear directorio de salida si no existe
    os.makedirs("motion_files", exist_ok=True)
    filepath = os.path.join("motion_files", filename)
    
    with open(filepath, 'w') as f:
        # Escribir encabezado en formato exacto de OpenSim
        f.write("Coordinates\n")
        f.write("version=1\n")
        f.write("nRows={}\n".format(len(self.motion_data)))
        f.write("nColumns=3\n")
        f.write("inDegrees=yes\n")
        f.write("\n")
        f.write("Units are S.I. units (second, meters, Newtons, ...)\n")
        f.write("Angles are in degrees.\n")
        f.write("\n")
        f.write("endheader\n")
        f.write("time\t{}\n".format('\t'.join(self.opensim_joints)))
        
        # Escribir datos con formato exacto de OpenSim
        for data_point in self.motion_data:
            f.write("{:.8f}\t".format(data_point['time']))
            for joint in self.opensim_joints:
                f.write("{:.8f}\t".format(data_point['angles'][joint]))
            f.write("\n")
```

### 6. Procesamiento de Frames: `process_frame()`

**Detección y Visualización:**
```python
def process_frame(self, frame, results):
    """Procesa un frame para detectar movimiento del brazo"""
    # Dibujar landmarks
    if results.pose_landmarks:
        self.mp_drawing.draw_landmarks(
            frame, results.pose_landmarks, self.mp_pose.POSE_CONNECTIONS)
        
        # Calcular ángulos del brazo
        angles = self.calculate_arm_angles(results.pose_landmarks.landmark)
        
        if angles:
            # Mostrar ángulos en pantalla
            cv2.putText(frame, f"Hombro: {angles['r_shoulder_elev']:.0f}°", 
                      (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(frame, f"Codo: {angles['r_elbow_flex']:.0f}°", 
                      (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
```

### 7. Bucle Principal: `run()`

**Inicialización de Cámara:**
```python
def run(self):
    """Ejecuta el sistema principal de captura"""
    # Inicializar cámara
    self.cap = cv2.VideoCapture(0)
    if not self.cap.isOpened():
        print("❌ Error: No se puede acceder a la cámara")
        return
    
    # Configurar resolución
    self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
```

**Bucle Principal:**
```python
while True:
    ret, frame = self.cap.read()
    if not ret:
        print("❌ Error leyendo frame de la cámara")
        break
    
    # Convertir BGR a RGB y procesar con MediaPipe
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = self.pose.process(rgb_frame)
    
    # Procesar frame
    processed_frame = self.process_frame(frame, results)
    
    # Mostrar frame
    cv2.imshow('Captura de Movimiento del Brazo - OpenSim Compatible', processed_frame)
    
    # Control de teclado
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
    elif key == ord('c'):
        # Calibrar posición inicial
        if results.pose_landmarks:
            self.calibrate_initial_position(results.pose_landmarks.landmark)
    elif key == ord('r'):
        if self.recording:
            self.stop_recording()
        else:
            if not self.calibrated:
                print("⚠️ Calibra primero la posición inicial con 'c'")
            else:
                self.start_recording()
    elif key == ord('s'):
        self.save_motion_file()
```

## 📁 Estructura de Archivos

```
opensim/
├── main.py              # Archivo principal del sistema
├── requirements.txt     # Dependencias del proyecto
├── README.md           # Documentación del proyecto
└── motion_files/       # Directorio de archivos .mot generados
    ├── arm26_motion_YYYYMMDD_HHMMSS.mot
    └── final_motion.mot
```

## 🔧 Configuración Avanzada

### Ajuste de Parámetros de MediaPipe

```python
self.pose = self.mp_pose.Pose(
    static_image_mode=False,        # Modo de video en tiempo real
    model_complexity=1,             # Complejidad del modelo (0-2)
    enable_segmentation=False,      # Segmentación habilitada/deshabilitada
    min_detection_confidence=0.8,   # Confianza mínima para detección
    min_tracking_confidence=0.6     # Confianza mínima para seguimiento
)
```

### Ajuste de Resolución de Cámara

```python
# Resolución estándar
self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

# Resolución HD (opcional)
self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
```

## 🐛 Solución de Problemas

### Problemas Comunes:

1. **Error de cámara**: Verificar que la cámara esté conectada y no esté siendo usada por otra aplicación
2. **Detección pobre**: Ajustar la iluminación y posición del brazo
3. **Ángulos incorrectos**: Recalibrar la posición inicial con la tecla 'c'
4. **Archivos no guardados**: Verificar permisos de escritura en el directorio

### Mensajes de Error:

- `❌ Error: No se puede acceder a la cámara`: Cámara no disponible
- `⚠️ Calibra primero la posición inicial con 'c'`: Necesario calibrar antes de grabar
- `❌ Error guardando archivo`: Problema de permisos o espacio en disco

## 📈 Mejoras Futuras

- [ ] Soporte para múltiples brazos (izquierdo y derecho)
- [ ] Calibración automática basada en IA
- [ ] Exportación a otros formatos (CSV, JSON)
- [ ] Interfaz gráfica con tkinter o PyQt
- [ ] Análisis de datos en tiempo real
- [ ] Integración directa con OpenSim

## 🤝 Contribuciones

Para contribuir al proyecto:

1. Fork del repositorio
2. Crear una rama para la nueva funcionalidad
3. Realizar los cambios
4. Crear un Pull Request

## 📄 Licencia

Este proyecto está bajo la Licencia MIT. Ver el archivo LICENSE para más detalles.

## 👥 Autores

- **Desarrollador Principal**: [Tu Nombre]
- **Fecha de Creación**: [Fecha]
- **Versión**: 1.0.0

## 📞 Soporte

Para soporte técnico o preguntas sobre el proyecto, contactar a través de:
- Email: [tu-email@ejemplo.com]
- GitHub Issues: [enlace al repositorio]

---

**Nota**: Este sistema está diseñado específicamente para captura de movimiento del brazo derecho. Para usar con el brazo izquierdo, se requieren modificaciones en el código.
