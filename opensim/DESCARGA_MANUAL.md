# Instrucciones de Descarga Manual de Modelos OpenPose

Los archivos actuales están corruptos. Sigue estas instrucciones para descargarlos manualmente:

## Archivo 1: pose_deploy_linevec.prototxt

### Opción A - GitHub (Recomendado)
1. Abre tu navegador
2. Visita: https://raw.githubusercontent.com/CMU-Perceptual-Computing-Lab/openpose/master/models/pose/coco/pose_deploy_linevec.prototxt
3. Presiona `Ctrl+S` para guardar
4. Guárdalo como: `pose_deploy_linevec.prototxt`
5. Muévelo a: `models/pose/coco/pose_deploy_linevec.prototxt`

### Opción B - Alternativa
1. Visita: https://github.com/opencv/opencv_extra/blob/master/testdata/dnn/openpose_pose_coco.prototxt
2. Haz clic en "Raw"
3. Guarda el archivo
4. Renómbralo a `pose_deploy_linevec.prototxt`
5. Muévelo a: `models/pose/coco/pose_deploy_linevec.prototxt`

## Archivo 2: pose_iter_440000.caffemodel (~200 MB)

### Opción A - Google Drive (Más fácil)
1. Visita: https://drive.google.com/file/d/1XISgkmF6kpNCfQ4vfRj-qLwpzKmjLgWf/view
2. Haz clic en "Descargar"
3. Espera a que se complete la descarga (~200 MB)
4. Muévelo a: `models/pose/coco/pose_iter_440000.caffemodel`

### Opción B - Sitio Oficial CMU
1. Visita: http://posefs1.perception.cs.cmu.edu/OpenPose/models/pose/coco/
2. Descarga: `pose_iter_440000.caffemodel`
3. Muévelo a: `models/pose/coco/pose_iter_440000.caffemodel`

### Opción C - Dropbox
1. Visita: https://www.dropbox.com/s/2dw1oz9t6hkx7g8/pose_iter_440000.caffemodel?dl=1
2. La descarga debería iniciar automáticamente
3. Muévelo a: `models/pose/coco/pose_iter_440000.caffemodel`

## Verificar Descarga

Después de descargar los archivos, ejecuta:

```bash
python verify_models.py
```

Deberías ver:
```
pose_deploy_linevec.prototxt: 0.05 MB [OK]
pose_iter_440000.caffemodel: 200.23 MB [OK]
```

## Estructura Final

Tu directorio debe verse así:

```
opensim/
├── models/
│   └── pose/
│       └── coco/
│           ├── pose_deploy_linevec.prototxt  (~50 KB)
│           └── pose_iter_440000.caffemodel   (~200 MB)
├── main.py
├── openpose.py
└── download_models.py
```

## Ejecutar OpenPose

Una vez verificados los modelos:

```bash
python openpose.py
```

## Alternativa: Usar MediaPipe

Si tienes problemas con OpenPose, puedes usar la versión con MediaPipe que es más fácil de configurar:

```bash
python main.py
```

MediaPipe no requiere descargas manuales y funciona inmediatamente después de instalar las dependencias.

## Problemas Comunes

### Error: "HTTP Error 429"
- El servidor de GitHub está limitando las peticiones
- Espera unos minutos e intenta nuevamente
- Usa una URL alternativa

### Error: "getaddrinfo failed"
- Problema de conexión a internet
- Verifica tu conexión
- Intenta con una VPN si está bloqueado

### Error en OpenCV DNN
- Los archivos están corruptos o incompletos
- Elimina los archivos corruptos
- Descarga nuevamente desde las URLs alternativas

## Soporte

Si continúas teniendo problemas:
1. Verifica que los archivos tengan el tamaño correcto
2. Asegúrate de que estén en el directorio correcto
3. Usa la versión MediaPipe como alternativa (`python main.py`)

