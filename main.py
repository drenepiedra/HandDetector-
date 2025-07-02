import mediapipe as mp
import cv2

# Configurar captura de video
dispositivoCapture = cv2.VideoCapture(0)

# Inicializar MediaPipe Hands correctamente
mpManos = mp.solutions.hands
manos = mpManos.Hands(
    static_image_mode=False,
    max_num_hands=2,
    min_detection_confidence=0.9,
    min_tracking_confidence=0.8,
)

# Utilidad para dibujar
mpDibujar = mp.solutions.drawing_utils

# Bucle principal
while True:
    success, img = dispositivoCapture.read()
    if not success:
        break

    # Convertir imagen a RGB
    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # Procesar imagen con MediaPipe
    resultado = manos.process(imgRGB)

    # Dibujar resultados
    if resultado.multi_hand_landmarks:
        for handLms in resultado.multi_hand_landmarks:
            mpDibujar.draw_landmarks(img, handLms, mpManos.HAND_CONNECTIONS)

    # Mostrar imagen
    cv2.imshow("Image", img)

    # Salir con tecla ESC
    if cv2.waitKey(1) & 0xFF == 27:
        break

# Liberar recursos
dispositivoCapture.release()
cv2.destroyAllWindows()
