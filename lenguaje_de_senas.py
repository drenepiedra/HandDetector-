import cv2
import mediapipe as mp
import numpy as np

def Etiqueta(idx, mano, results):
    aux = None
    for _, clase in enumerate(results.multi_handedness):
      if clase.classification[0].index == idx:
        label = clase.classification[0].label
        texto = '{}'.format(label)

        coords = tuple(np.multiply(np.array(
           (mano.landmark[mp_hands.HandLandmark.WRIST].x, 
            mano.landmark[mp_hands.HandLandmark.WRIST].y)),
            [1920, 1080]).astype(int))
        
        aux = texto, coords
    return aux

def distancia_euclidiana(p1, p2):
    d = ((p2[0] - p1[0]) ** 2 + (p2[1] - p1[1]) ** 2) ** 0.5
    return d

mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_hands = mp.solutions.hands
change = True
change2 = False

cap = cv2.VideoCapture(0)
with mp_hands.Hands(
    model_complexity=1,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7,
    max_num_hands=2) as hands:
  while cap.isOpened():
    success, image = cap.read()
    if not success:
      continue

    image.flags.writeable = False
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = hands.process(image)

    image.flags.writeable = True
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    image_height, image_width, _ = image.shape
    if results.multi_hand_landmarks:
        if len(results.multi_hand_landmarks):
            for num, hand_landmarks in enumerate(results.multi_hand_landmarks):
                
                mp_drawing.draw_landmarks(
                    image,
                    hand_landmarks,
                    mp_hands.HAND_CONNECTIONS,
                    mp_drawing_styles.get_default_hand_landmarks_style(),
                    mp_drawing_styles.get_default_hand_connections_style())

                index_finger_tip = (int(hand_landmarks.landmark[8].x * image_width),
                                int(hand_landmarks.landmark[8].y * image_height))
                index_finger_pip = (int(hand_landmarks.landmark[6].x * image_width),
                                int(hand_landmarks.landmark[6].y * image_height))
                
                thumb_tip = (int(hand_landmarks.landmark[4].x * image_width),
                                int(hand_landmarks.landmark[4].y * image_height))
                thumb_pip = (int(hand_landmarks.landmark[2].x * image_width),
                                int(hand_landmarks.landmark[2].y * image_height))
                
                middle_finger_tip = (int(hand_landmarks.landmark[12].x * image_width),
                                int(hand_landmarks.landmark[12].y * image_height))
                
                middle_finger_pip = (int(hand_landmarks.landmark[10].x * image_width),
                                int(hand_landmarks.landmark[10].y * image_height))
                
                ring_finger_tip = (int(hand_landmarks.landmark[16].x * image_width),
                                int(hand_landmarks.landmark[16].y * image_height))
                ring_finger_pip = (int(hand_landmarks.landmark[14].x * image_width),
                                int(hand_landmarks.landmark[14].y * image_height))
                
                pinky_tip = (int(hand_landmarks.landmark[20].x * image_width),
                                int(hand_landmarks.landmark[20].y * image_height))
                pinky_pip = (int(hand_landmarks.landmark[18].x * image_width),
                                int(hand_landmarks.landmark[18].y * image_height))
                
                wrist = (int(hand_landmarks.landmark[0].x * image_width),
                                int(hand_landmarks.landmark[0].y * image_height))
                
                # print(ring_finger_pip)
                # print(ring_finger_tip)
                # print(distancia_euclidiana(ring_finger_pip, ring_finger_tip))
                if thumb_pip[1] - thumb_tip[1] > 0 and thumb_pip[1] - index_finger_tip[1] < 0 \
                    and thumb_pip[1] - middle_finger_tip[1] < 0 and thumb_pip[1] - ring_finger_tip[1]<0 \
                    and thumb_pip[1] - pinky_tip[1] < 0:
                    cv2.putText(image, 'GOOD', (700, 150), 
                                cv2.FONT_HERSHEY_SIMPLEX, 
                                3.0, (0, 0, 255), 6) 
                elif thumb_pip[1] - thumb_tip[1] < 0 and thumb_pip[1] - index_finger_tip[1] > 0 \
                    and thumb_pip[1] - middle_finger_tip[1] > 0 and thumb_pip[1] - ring_finger_tip[1]>0 \
                    and thumb_pip[1] - pinky_tip[1] > 0:
                    cv2.putText(image, 'BAD', (700, 150), 
                                cv2.FONT_HERSHEY_SIMPLEX, 
                                3.0, (0, 0, 255), 6)
                   
                elif thumb_pip[1] - thumb_tip[1] > 0 and index_finger_pip[1] - index_finger_tip[1]>0 \
                    and pinky_pip[1] - pinky_tip[1] > 0:
                    cv2.putText(image, 'I LOVE YOU', (700, 150), 
                                cv2.FONT_HERSHEY_SIMPLEX, 
                                3.0, (0, 0, 255), 6)
                   
                    
                if Etiqueta(num, hand_landmarks, results) and len(results.multi_hand_landmarks)==2:
                    text,coords = Etiqueta(num, hand_landmarks, results)
                    print(text, coords)
                    if text =="Right":
                      #text = "IZQUIERDA"
                      index_finger_tip_r = (int(hand_landmarks.landmark[8].x * image_width),
                                int(hand_landmarks.landmark[8].y * image_height))
                      #print(index_finger_tip)
                      change = True
                    if text =="Left":
                        #text = "DERECHA"
                        index_finger_tip_l = (int(hand_landmarks.landmark[8].x * image_width),
                                int(hand_landmarks.landmark[8].y * image_height))
                      
                        wrist = (int(hand_landmarks.landmark[0].x * image_width),
                                int(hand_landmarks.landmark[0].y * image_height))

                        change2 = True

                    if change2 == True and change == True:
                        if distancia_euclidiana(index_finger_tip_l,  wrist) < 170.0:
                            cv2.putText(image, 'What time is it?', (700, 150), 
                                cv2.FONT_HERSHEY_SIMPLEX, 
                                3.0, (0, 0, 255), 6)
                    
                    cv2.putText(image, text, coords, cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255),2,cv2.LINE_AA)

                
    cv2.imshow('MediaPipe Hands', image)
    if cv2.waitKey(5) & 0xFF == 27:
      break
cap.release()
cv2.destroyAllWindows()