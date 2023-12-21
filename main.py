import cv2
import numpy as np

# Cargar el modelo YOLO
net = cv2.dnn.readNet("yolov3.weights", "yolov3.cfg")
classes = []
with open("coco.names", "r") as f:
    classes = [line.strip() for line in f.readlines()]

# Configurar la captura de video desde la webcam
cap = cv2.VideoCapture(0)

while True:
    # Leer el frame de la cámara
    ret, frame = cap.read()

    # Obtener las dimensiones del frame
    height, width, _ = frame.shape

    # Convertir el frame a un blob para la entrada de la red neuronal
    blob = cv2.dnn.blobFromImage(frame, 0.00392, (416, 416), (0, 0, 0), True, crop=False)

    # Establecer la entrada del modelo
    net.setInput(blob)

    # Obtener las capas de salida
    outs = net.forward(net.getUnconnectedOutLayersNames())

    # Inicializar listas de detecciones, confianzas y clases
    class_ids = []
    confidences = []
    boxes = []

    # Analizar las salidas de la red neuronal
    for out in outs:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > 0.5 and class_id == 2:  # 2 es el índice para "fire" en el modelo COCO
                center_x = int(detection[0] * width)
                center_y = int(detection[1] * height)
                w = int(detection[2] * width)
                h = int(detection[3] * height)

                # Calcular las coordenadas de la caja delimitadora
                x = int(center_x - w / 2)
                y = int(center_y - h / 2)

                # Agregar las coordenadas, confianza y clase a las listas respectivas
                boxes.append([x, y, w, h])
                confidences.append(float(confidence))
                class_ids.append(class_id)

    # Aplicar la supresión no máxima para eliminar detecciones redundantes
    indices = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)

    for i in range(len(boxes)):
        if i in indices:
            x, y, w, h = boxes[i]
            label = str(classes[class_ids[i]])
            confidence = confidences[i]
            color = (0, 255, 0)  # Color verde para las detecciones de incendios
            cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
            cv2.putText(frame, f"{label} {confidence:.2f}", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    # Mostrar el frame resultante
    cv2.imshow("Fire Detection", frame)

    # Salir del bucle con la tecla 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Liberar los recursos
cap.release()
cv2.destroyAllWindows()
