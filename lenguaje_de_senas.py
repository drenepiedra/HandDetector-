import cv2
import mediapipe as mp
import numpy as np
import time

# Inicializar MediaPipe
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

# === FUNCIÓN DE DETECCIÓN DE LETRAS ===
def detectar_letra(hand_landmarks):
    dedos_arriba = []
    tips = [8, 12, 16, 20]  # índice, medio, anular, meñique

    for tip in tips:
        if hand_landmarks.landmark[tip].y < hand_landmarks.landmark[tip - 2].y:
            dedos_arriba.append(True)
        else:
            dedos_arriba.append(False)

    pulgar_x = hand_landmarks.landmark[4].x
    pulgar_y = hand_landmarks.landmark[4].y

    # LETRA A
    if not any(dedos_arriba):
        return "A"

    # LETRA E
    if not any(dedos_arriba) and pulgar_y > hand_landmarks.landmark[8].y:
        return "E"

    # LETRA M
    if not dedos_arriba[0] and not dedos_arriba[1] and not dedos_arriba[2] and dedos_arriba[3]:
        return "M"

    # LETRA T
    if not any(dedos_arriba) and pulgar_x > hand_landmarks.landmark[3].x:
        return "T"

    # LETRA O
    if all([
        hand_landmarks.landmark[8].x < hand_landmarks.landmark[7].x,
        hand_landmarks.landmark[12].x < hand_landmarks.landmark[11].x,
        hand_landmarks.landmark[16].x < hand_landmarks.landmark[15].x,
        hand_landmarks.landmark[20].x < hand_landmarks.landmark[19].x,
    ]):
        return "O"

    # LETRA U
    if dedos_arriba[0] and dedos_arriba[1] and not dedos_arriba[2] and not dedos_arriba[3]:
        return "U"

    # LETRA C
    if all(dedos_arriba) and (
        hand_landmarks.landmark[4].x < hand_landmarks.landmark[8].x and
        hand_landmarks.landmark[20].x > hand_landmarks.landmark[16].x
    ):
        return "C"

    # LETRA H
    if dedos_arriba[0] and dedos_arriba[1] and not dedos_arriba[2] and not dedos_arriba[3]:
        if hand_landmarks.landmark[8].y > hand_landmarks.landmark[6].y:
            return "H"

    # LETRA I
    if not dedos_arriba[0] and not dedos_arriba[1] and not dedos_arriba[2] and dedos_arriba[3]:
        return "I"

    # LETRA R
    if dedos_arriba[0] and dedos_arriba[1] and not dedos_arriba[2] and not dedos_arriba[3]:
        if hand_landmarks.landmark[8].x > hand_landmarks.landmark[12].x:
            return "R"

    return None

# === VARIABLES PARA DETECCIÓN Y MENSAJE ===
mensaje = ""
ultima_letra = ""
tiempo_ultima = time.time()

# === CONFIGURAR CÁMARA ===
cam = cv2.VideoCapture(0)
hands = mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.8)

while True:
    ret, frame = cam.read()
    frame = cv2.flip(frame, 1)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb)

    altura, ancho, _ = frame.shape

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            letra = detectar_letra(hand_landmarks)

            if letra and letra != ultima_letra and time.time() - tiempo_ultima > 1.5:
                mensaje += letra
                ultima_letra = letra
                tiempo_ultima = time.time()

            if letra:
                cv2.putText(frame, f"Letra: {letra}", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 2)

    # Zona blanca para el mensaje acumulado
    zona_blanca = 255 * np.ones((100, ancho, 3), dtype=np.uint8)
    cv2.putText(zona_blanca, mensaje, (30, 70), cv2.FONT_HERSHEY_COMPLEX, 2, (0, 0, 0), 3)

    # Unir zona blanca con el frame
    frame_completo = cv2.vconcat([frame, zona_blanca])

    cv2.imshow("Lenguaje de Señas - Detector", frame_completo)

    tecla = cv2.waitKey(1) & 0xFF
    if tecla == 27:  # ESC para salir
        break
    elif tecla == ord(' '):  # Espacio
        mensaje += " "
    elif tecla == ord('b'):  # Borrar letra
        mensaje = mensaje[:-1]

cam.release()
cv2.destroyAllWindows()
