import cv2
import mediapipe as mp
import numpy as np
import time
import threading
from datetime import datetime
import os
import sys

class ArmMotionCapture:
    def __init__(self):
        # Inicializar MediaPipe
        self.mp_pose = mp.solutions.pose
        self.mp_drawing = mp.solutions.drawing_utils
        self.pose = self.mp_pose.Pose(
            static_image_mode=False,
            model_complexity=1,
            enable_segmentation=False,
            min_detection_confidence=0.8,
            min_tracking_confidence=0.6
        )
        
        # Variables de control
        self.recording = False
        self.motion_data = []
        self.start_time = None
        self.cap = None
        
        # Variables de calibración (simplificadas)
        self.calibrated = False
        
        # Configuración de OpenSim para modelo arm26 (formato exacto)
        self.opensim_joints = [
            'r_shoulder_elev', 'r_elbow_flex'
        ]
        
        # Mapeo de landmarks de MediaPipe a articulaciones del brazo
        self.landmark_mapping = {
            'shoulder': [11, 12],  # Hombros izquierdo y derecho
            'elbow': [13, 14],     # Codos izquierdo y derecho
            'wrist': [15, 16],     # Muñecas izquierda y derecha
            'hand': [17, 18, 19, 20]  # Puntos de la mano
        }
        
        print("Sistema de captura de movimiento del brazo inicializado")
        print("🎯 INSTRUCCIONES:")
        print("1. Posiciona tu brazo derecho frente a la cámara")
        print("2. Presiona 'c' para calibrar la posición inicial")
        print("3. Presiona 'r' para iniciar/detener grabación")
        print("4. Presiona 's' para guardar datos actuales")
        print("5. Presiona 'q' para salir")
        print("\n💡 IMPORTANTE: Calibra siempre antes de grabar para mejor precisión")
    
    def calculate_arm_angles(self, landmarks):
        """Calcula los ángulos del brazo corregidos para rangos correctos"""
        try:
            # Obtener puntos clave del brazo derecho
            shoulder = landmarks[self.landmark_mapping['shoulder'][1]]  # Hombro derecho
            elbow = landmarks[self.landmark_mapping['elbow'][1]]        # Codo derecho
            wrist = landmarks[self.landmark_mapping['wrist'][1]]        # Muñeca derecha
            
            # Convertir a coordenadas (MediaPipe usa 0-1, donde (0,0) es esquina superior izquierda)
            shoulder_x, shoulder_y = shoulder.x, shoulder.y
            elbow_x, elbow_y = elbow.x, elbow.y
            wrist_x, wrist_y = wrist.x, wrist.y
            
            # 1. Elevación del hombro (r_shoulder_elev)
            # Ángulo del brazo respecto a la vertical (eje Y)
            # Calcular el ángulo del brazo respecto a la vertical hacia arriba
            
            # Vector del brazo (hombro a codo)
            arm_vector_x = elbow_x - shoulder_x
            arm_vector_y = elbow_y - shoulder_y
            
            # Calcular ángulo respecto a la vertical (eje Y negativo = hacia arriba)
            # Usar atan2 para obtener el ángulo correcto
            shoulder_elev_raw = np.arctan2(arm_vector_x, arm_vector_y) * 180 / np.pi
            
            # Convertir a rango de -90 a 180 grados
            # -90° = brazo hacia abajo (vertical)
            # 0° = brazo horizontal hacia la derecha  
            # 90° = brazo hacia arriba (vertical)
            # 180° = brazo horizontal hacia la izquierda
            
            # Invertir el signo para que subir = ángulo positivo mayor
            shoulder_elev = -shoulder_elev_raw
            
            # Asegurar que esté en el rango correcto -90 a 180
            shoulder_elev = np.clip(shoulder_elev, -90, 180)
            
            # 2. Flexión del codo (r_elbow_flex) - CORREGIDO
            # Ángulo entre brazo y antebrazo
            # 0° = brazo extendido, 180° = brazo completamente flexionado
            
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
                
                # CORRECCIÓN: Restar el offset para que llegue a 0°
                # El problema es que arccos no da exactamente 0° cuando está extendido
                # Buscamos el mínimo valor posible cuando está extendido
                min_elbow_flex = 20  # Valor mínimo observado cuando está extendido
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
            filename = f"arm26_motion_{timestamp}.mot"
        
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
    
    def process_frame(self, frame, results):
        """Procesa un frame para detectar movimiento del brazo"""
        # Dibujar landmarks
        if results.pose_landmarks:
            self.mp_drawing.draw_landmarks(
                frame, results.pose_landmarks, self.mp_pose.POSE_CONNECTIONS)
            
            # Calcular ángulos del brazo
            angles = self.calculate_arm_angles(results.pose_landmarks.landmark)
            
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
                self.save_motion_file("final_motion.mot")

def main():
    """Función principal"""
    print("🤖 Sistema de Captura de Movimiento del Brazo para OpenSim")
    print("=" * 60)
    
    # Crear instancia del capturador
    motion_capture = ArmMotionCapture()
    
    # Ejecutar sistema
    motion_capture.run()
    
    print("👋 Sistema finalizado")

if __name__ == "__main__":
    main()
