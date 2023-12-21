# Detección de Incendios con OpenCV y YOLO

Este proyecto implementa un sistema de detección de incendios utilizando OpenCV y el modelo YOLOv3. La detección se realiza en tiempo real a través de diferentes fuentes, en este caso, a través de la cámara web, gracias al framework GStream.

## Requisitos

- Python 3.x
- OpenCV
- Modelo YOLOv3 entrenado para la detección de objetos (pesos, configuración y nombres de clases)
  
## Instalación de Dependencias

pip install opencv-python

## Uso

- Descarga el modelo YOLOv3 y los archivos de configuración desde el sitio oficial de YOLO (https://pjreddie.com/darknet/yolo/).
- Coloca los archivos de modelo, configuración y nombres de clases en el directorio del proyecto.
- Ejecuta el script main.py.
