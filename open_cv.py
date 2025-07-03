import numpy as np
import cv2

cap = cv2.VideoCapture(1)

while (cap.isOpened()):
    # Capture frame-by-frame
    ret, frame = cap.read()

    if ret:
        # Display the resulting frame
        cv2.imshow('frame', frame)
    
        # Exit?
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    else:
        break

cap.release()
cv2.destroyAllWindows()