import cv2
import numpy as np
import time
import os
from datetime import datetime

class ArmMotionCaptureOpenPose:
    def __init__(self, openpose_model_path):
        """
        Inicializa el sistema de captura de movimiento con OpenPose
        
        Args:
            openpose_model_path: Ruta al directorio de modelos de OpenPose
        """
        # Configuración de OpenPose
        self.model_path = openpose_model_path
        self.proto_file = os.path.join(openpose_model_path, "pose_deploy_linevec.prototxt")
        self.weights_file = os.path.join(openpose_model_path, "pose_iter_440000.caffemodel")
        
        # Cargar red neuronal de OpenPose
        try:
            self.net = cv2.dnn.readNetFromCaffe(self.proto_file, self.weights_file)
            print("✅ Modelo OpenPose cargado correctamente")
        except Exception as e:
            print(f"❌ Error cargando modelo OpenPose: {e}")
            print("💡 Descarga los modelos desde: https://github.com/CMU-Perceptual-Computing-Lab/openpose")
            raise
        
        # Variables de control
        self.recording = False
        self.motion_data = []
        self.start_time = None
        self.cap = None
        self.calibrated = False
        
        # Configuración de OpenSim para modelo arm26
        self.opensim_joints = [
            'r_shoulder_elev', 'r_elbow_flex'
        ]
        
        # Mapeo de puntos clave de OpenPose (modelo BODY_25)
        # https://github.com/CMU-Perceptual-Computing-Lab/openpose/blob/master/doc/output.md
        self.POSE_PAIRS = {
            'Neck': 1,
            'RShoulder': 2,  # Hombro derecho
            'RElbow': 3,     # Codo derecho
            'RWrist': 4,     # Muñeca derecha
            'LShoulder': 5,  # Hombro izquierdo
            'LElbow': 6,     # Codo izquierdo
            'LWrist': 7,     # Muñeca izquierda
        }
        
        # Umbral de confianza para detección
        self.threshold = 0.1
        
        # Dimensiones de entrada para OpenPose
        self.inWidth = 368
        self.inHeight = 368
        
        print("🤖 Sistema de captura de movimiento del brazo con OpenPose inicializado")
        print("🎯 INSTRUCCIONES:")
        print("1. Posiciona tu brazo derecho frente a la cámara")
        print("2. Presiona 'c' para calibrar la posición inicial")
        print("3. Presiona 'r' para iniciar/detener grabación")
        print("4. Presiona 's' para guardar datos actuales")
        print("5. Presiona 'q' para salir")
        print("\n💡 IMPORTANTE: Calibra siempre antes de grabar para mejor precisión")
    
    def detect_pose(self, frame):
        """
        Detecta la pose en el frame usando OpenPose
        
        Args:
            frame: Frame de video
            
        Returns:
            points: Diccionario con las coordenadas de los puntos clave detectados
        """
        frameWidth = frame.shape[1]
        frameHeight = frame.shape[0]
        
        # Preparar la entrada para la red
        inpBlob = cv2.dnn.blobFromImage(
            frame, 
            1.0 / 255, 
            (self.inWidth, self.inHeight),
            (0, 0, 0), 
            swapRB=False, 
            crop=False
        )
        
        self.net.setInput(inpBlob)
        
        # Obtener la salida de la red
        output = self.net.forward()
        
        H = output.shape[2]
        W = output.shape[3]
        
        # Diccionario para almacenar puntos detectados
        points = {}
        
        # Extraer puntos clave
        for name, idx in self.POSE_PAIRS.items():
            # Mapa de confianza para el punto clave
            probMap = output[0, idx, :, :]
            
            # Encontrar el máximo global
            minVal, prob, minLoc, point = cv2.minMaxLoc(probMap)
            
            # Escalar el punto a las dimensiones originales del frame
            x = (frameWidth * point[0]) / W
            y = (frameHeight * point[1]) / H
            
            # Si la confianza es mayor al umbral, guardar el punto
            if prob > self.threshold:
                points[name] = (int(x), int(y), prob)
            else:
                points[name] = None
        
        return points
    
    def calculate_arm_angles(self, points):
        """
        Calcula los ángulos del brazo a partir de los puntos detectados
        
        Args:
            points: Diccionario con las coordenadas de los puntos clave
            
        Returns:
            dict: Diccionario con los ángulos calculados
        """
        try:
            # Verificar que se detectaron todos los puntos necesarios
            if points['RShoulder'] is None or points['RElbow'] is None or points['RWrist'] is None:
                return None
            
            # Obtener coordenadas
            shoulder_x, shoulder_y, _ = points['RShoulder']
            elbow_x, elbow_y, _ = points['RElbow']
            wrist_x, wrist_y, _ = points['RWrist']
            
            # Normalizar coordenadas (0-1)
            # Nota: OpenPose ya da coordenadas en píxeles, necesitamos normalizarlas
            # para que sean comparables con el sistema MediaPipe
            
            # 1. Elevación del hombro (r_shoulder_elev)
            # Vector del brazo (hombro a codo)
            arm_vector_x = elbow_x - shoulder_x
            arm_vector_y = elbow_y - shoulder_y
            
            # Calcular ángulo respecto a la vertical
            shoulder_elev_raw = np.arctan2(arm_vector_x, arm_vector_y) * 180 / np.pi
            
            # Invertir el signo para que subir = ángulo positivo mayor
            shoulder_elev = -shoulder_elev_raw
            
            # Asegurar que esté en el rango correcto -90 a 180
            shoulder_elev = np.clip(shoulder_elev, -90, 180)
            
            # 2. Flexión del codo (r_elbow_flex)
            # Vector del brazo (hombro a codo)
            upper_arm_x = elbow_x - shoulder_x
            upper_arm_y = elbow_y - shoulder_y
            
            # Vector del antebrazo (codo a muñeca)
            forearm_x = wrist_x - elbow_x
            forearm_y = wrist_y - elbow_y
            
            # Calcular ángulo entre vectores
            dot_product = upper_arm_x * forearm_x + upper_arm_y * forearm_y
            upper_arm_length = np.sqrt(upper_arm_x**2 + upper_arm_y**2)
            forearm_length = np.sqrt(forearm_x**2 + forearm_y**2)
            
            if upper_arm_length > 0 and forearm_length > 0:
                cos_angle = dot_product / (upper_arm_length * forearm_length)
                cos_angle = np.clip(cos_angle, -1.0, 1.0)
                elbow_flex = np.arccos(cos_angle) * 180 / np.pi
                
                # Corrección para que llegue a 0° cuando está extendido
                min_elbow_flex = 20
                elbow_flex = max(0, elbow_flex - min_elbow_flex)
            else:
                elbow_flex = 0
            
            return {
                'r_shoulder_elev': shoulder_elev,
                'r_elbow_flex': elbow_flex
            }
            
        except Exception as e:
            print(f"Error calculando ángulos: {e}")
            return None
    
    def draw_skeleton(self, frame, points):
        """
        Dibuja el esqueleto detectado en el frame
        
        Args:
            frame: Frame de video
            points: Diccionario con las coordenadas de los puntos clave
        """
        # Dibujar puntos clave del brazo derecho
        if points['RShoulder'] is not None:
            cv2.circle(frame, (points['RShoulder'][0], points['RShoulder'][1]), 8, (0, 255, 255), -1)
            cv2.putText(frame, "Hombro", (points['RShoulder'][0] + 10, points['RShoulder'][1]), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        if points['RElbow'] is not None:
            cv2.circle(frame, (points['RElbow'][0], points['RElbow'][1]), 8, (0, 255, 255), -1)
            cv2.putText(frame, "Codo", (points['RElbow'][0] + 10, points['RElbow'][1]), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        if points['RWrist'] is not None:
            cv2.circle(frame, (points['RWrist'][0], points['RWrist'][1]), 8, (0, 255, 255), -1)
            cv2.putText(frame, "Muneca", (points['RWrist'][0] + 10, points['RWrist'][1]), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        # Dibujar líneas conectando los puntos
        if points['RShoulder'] is not None and points['RElbow'] is not None:
            cv2.line(frame, 
                    (points['RShoulder'][0], points['RShoulder'][1]),
                    (points['RElbow'][0], points['RElbow'][1]),
                    (0, 255, 0), 3)
        
        if points['RElbow'] is not None and points['RWrist'] is not None:
            cv2.line(frame, 
                    (points['RElbow'][0], points['RElbow'][1]),
                    (points['RWrist'][0], points['RWrist'][1]),
                    (0, 255, 0), 3)
    
    def calibrate_initial_position(self, points):
        """Calibra la posición inicial del brazo"""
        try:
            angles = self.calculate_arm_angles(points)
            if angles:
                self.calibrated = True
                print("✅ Calibración completada")
                print(f"   Elevación del hombro: {angles['r_shoulder_elev']:.1f}°")
                print(f"   Flexión del codo: {angles['r_elbow_flex']:.1f}°")
                return True
        except Exception as e:
            print(f"Error en calibración: {e}")
        return False
    
    def start_recording(self):
        """Inicia la grabación de movimiento"""
        if not self.recording:
            self.recording = True
            self.motion_data = []
            self.start_time = time.time()
            print("🔴 Grabación iniciada")
    
    def stop_recording(self):
        """Detiene la grabación de movimiento"""
        if self.recording:
            self.recording = False
            print("⏹️ Grabación detenida")
    
    def save_motion_file(self, filename=None):
        """Guarda los datos de movimiento en formato .mot compatible con OpenSim arm26"""
        if not self.motion_data:
            print("No hay datos para guardar")
            return
        
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"arm26_motion_openpose_{timestamp}.mot"
        
        # Crear directorio de salida si no existe
        os.makedirs("motion_files", exist_ok=True)
        filepath = os.path.join("motion_files", filename)
        
        try:
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
            
            print(f"✅ Archivo guardado: {filepath}")
            print(f"📊 Datos guardados: {len(self.motion_data)} puntos")
            print(f"🎯 Formato: OpenSim arm26 compatible")
            
        except Exception as e:
            print(f"❌ Error guardando archivo: {e}")
    
    def process_frame(self, frame, points):
        """Procesa un frame para detectar movimiento del brazo"""
        # Dibujar esqueleto
        self.draw_skeleton(frame, points)
        
        # Calcular ángulos del brazo
        angles = self.calculate_arm_angles(points)
        
        if angles:
            # Mostrar ángulos en pantalla de forma simple
            cv2.putText(frame, f"Hombro: {angles['r_shoulder_elev']:.0f}°", 
                      (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(frame, f"Codo: {angles['r_elbow_flex']:.0f}°", 
                      (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            # Mostrar estado de calibración
            if not self.calibrated:
                cv2.putText(frame, "Presiona 'c' para calibrar", 
                          (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
            else:
                cv2.putText(frame, "Listo para grabar", 
                          (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
            
            # Agregar datos de movimiento si está grabando
            if self.recording:
                current_time = time.time() - self.start_time if self.start_time else 0
                motion_point = {
                    'time': current_time,
                    'angles': angles
                }
                self.motion_data.append(motion_point)
        else:
            cv2.putText(frame, "No se detecta el brazo", 
                      (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        
        # Mostrar estado de grabación
        status_text = "🔴 GRABANDO" if self.recording else "⏸️ PAUSADO"
        cv2.putText(frame, status_text, (frame.shape[1] - 150, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255) if self.recording else (255, 0, 0), 2)
        
        # Mostrar instrucciones actualizadas
        instructions = "Controles: 'c'=calibrar, 'r'=grabar, 's'=guardar, 'q'=salir"
        cv2.putText(frame, instructions, (10, frame.shape[0] - 20), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        return frame
    
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
        
        print("📹 Cámara inicializada")
        print("🎯 Posiciona tu brazo derecho frente a la cámara")
        
        try:
            while True:
                ret, frame = self.cap.read()
                if not ret:
                    print("❌ Error leyendo frame de la cámara")
                    break
                
                # Detectar pose con OpenPose
                points = self.detect_pose(frame)
                
                # Procesar frame
                processed_frame = self.process_frame(frame, points)
                
                # Mostrar frame
                cv2.imshow('Captura de Movimiento del Brazo - OpenPose + OpenSim', processed_frame)
                
                # Control de teclado
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break
                elif key == ord('c'):
                    # Calibrar posición inicial
                    self.calibrate_initial_position(points)
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
        
        except KeyboardInterrupt:
            print("\n🛑 Interrumpido por el usuario")
        
        finally:
            # Limpiar recursos
            if self.cap:
                self.cap.release()
            cv2.destroyAllWindows()
            
            # Guardar datos si hay grabación activa
            if self.recording and self.motion_data:
                print("💾 Guardando datos finales...")
                self.save_motion_file("final_motion_openpose.mot")

def main():
    """Función principal"""
    print("🤖 Sistema de Captura de Movimiento del Brazo con OpenPose para OpenSim")
    print("=" * 70)
    
    # Ruta al directorio de modelos de OpenPose
    # NOTA: Debes descargar los modelos de OpenPose primero
    # Descarga desde: https://github.com/CMU-Perceptual-Computing-Lab/openpose/blob/master/models/getModels.sh
    openpose_model_path = "models/pose/coco"
    
    # Verificar que existe la ruta de modelos
    if not os.path.exists(openpose_model_path):
        print("⚠️ ADVERTENCIA: No se encontró el directorio de modelos de OpenPose")
        print(f"   Ruta esperada: {openpose_model_path}")
        print("\n📥 INSTRUCCIONES PARA DESCARGAR MODELOS:")
        print("1. Crea el directorio: mkdir -p models/pose/coco")
        print("2. Descarga los archivos:")
        print("   - pose_deploy_linevec.prototxt")
        print("   - pose_iter_440000.caffemodel")
        print("3. Desde: https://github.com/CMU-Perceptual-Computing-Lab/openpose")
        print("\nO usa esta ruta alternativa si tienes OpenPose instalado:")
        openpose_model_path = input("Ingresa la ruta a los modelos de OpenPose (o Enter para usar './models/pose/coco'): ").strip()
        if not openpose_model_path:
            openpose_model_path = "models/pose/coco"
    
    try:
        # Crear instancia del capturador
        motion_capture = ArmMotionCaptureOpenPose(openpose_model_path)
        
        # Ejecutar sistema
        motion_capture.run()
        
        print("👋 Sistema finalizado")
    except Exception as e:
        print(f"❌ Error: {e}")
        print("\n💡 Asegúrate de haber descargado los modelos de OpenPose correctamente")

if __name__ == "__main__":
    main()

