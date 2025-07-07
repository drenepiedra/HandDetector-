import cv2

def encontrar_webcam_activa(max_dispositivos=5):
    for i in range(max_dispositivos):
        cap = cv2.VideoCapture(i)
        if cap.read()[0]:
            print(f"✅ Cámara activa en el índice {i}")
            cap.release()
        else:
            print(f"❌ No hay cámara en el índice {i}")

encontrar_webcam_activa()
