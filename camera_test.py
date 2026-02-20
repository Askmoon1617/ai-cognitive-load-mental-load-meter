# camera.py
import cv2
import mediapipe as mp
import numpy as np

mp_face = mp.solutions.face_mesh
face_mesh = mp_face.FaceMesh(max_num_faces=1, refine_landmarks=True)

LEFT_EYE = [33, 160, 158, 133, 153, 144]
RIGHT_EYE = [362, 385, 387, 263, 373, 380]
FOREHEAD = [10, 338]

def ear_calc(eye, lm, w, h):
    pts = [(int(lm[i].x * w), int(lm[i].y * h)) for i in eye]
    A = np.linalg.norm(np.array(pts[1]) - np.array(pts[5]))
    B = np.linalg.norm(np.array(pts[2]) - np.array(pts[4]))
    C = np.linalg.norm(np.array(pts[0]) - np.array(pts[3]))
    return (A + B) / (2.0 * C)

def analyze_frame(frame):
    """
    Input  : BGR frame (numpy array)
    Output : score, annotated_frame, detected_region
    """
    h, w, _ = frame.shape
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(rgb)

    score = 35
    region = "Calm"

    if results.multi_face_landmarks:
        lm = results.multi_face_landmarks[0].landmark
        ear = (ear_calc(LEFT_EYE, lm, w, h) +
               ear_calc(RIGHT_EYE, lm, w, h)) / 2

        if ear < 0.20:
            score = 75
            region = "Eyes + Forehead (High Load)"
            color = (0, 0, 255)
        elif ear < 0.25:
            score = 55
            region = "Eyes (Moderate Load)"
            color = (0, 255, 255)
        else:
            score = 35
            region = "Relaxed State"
            color = (0, 255, 0)

        # Draw forehead line
        f1 = (int(lm[FOREHEAD[0]].x * w), int(lm[FOREHEAD[0]].y * h))
        f2 = (int(lm[FOREHEAD[1]].x * w), int(lm[FOREHEAD[1]].y * h))
        cv2.line(frame, f1, f2, color, 3)

        # Draw eye lines
        for eye in [LEFT_EYE, RIGHT_EYE]:
            p1 = (int(lm[eye[0]].x * w), int(lm[eye[0]].y * h))
            p2 = (int(lm[eye[3]].x * w), int(lm[eye[3]].y * h))
            cv2.line(frame, p1, p2, color, 2)

    return score, frame, region