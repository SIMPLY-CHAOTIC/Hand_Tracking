import mediapipe as mp
import cv2
import numpy as np

cap=cv2.VideoCapture(0)

hands=mp.solutions.hands
hands_mesh=hands.Hands(static_image_mode=True,min_detection_confidence=0.7)
draw=mp.solutions.drawing_utils

while True:
    _,frm=cap.read()
    rgb=cv2.cvtColor(frm,cv2.COLOR_BGR2RGB)
    op=hands_mesh.process(rgb)

    if op.multi_hand_landmarks:
        for i in op.multi_hand_landmarks:
            draw.draw_landmarks(frm,i,hands.HAND_CONNECTIONS,landmark_drawing_spec=draw.DrawingSpec(circle_radius=1))
    
    cv2.imshow("windows",frm)
    if cv2.waitKey(1)==27:
        cap.release()
        cv2.destroyAllWindows()
        break