import cv2
import numpy as np
import json
import dlib
import random
import time
import os

# ---------------- PATH SETUP ---------------- #

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Dlib landmark model
PREDICTOR_PATH = os.path.join(
    BASE_DIR,
    "shape_predictor_68_face_landmarks.dat"
)

# Trainer paths
TRAINER_DIR = os.path.join(BASE_DIR, "trainer")
TRAINER_PATH = os.path.join(TRAINER_DIR, "trainer.yml")

# User database
USERS_PATH = os.path.join(BASE_DIR, "database.json")

os.makedirs(TRAINER_DIR, exist_ok=True)

print("File exists:", os.path.exists(PREDICTOR_PATH))
print("Path:", PREDICTOR_PATH)


# ---------------- RECOGNIZER ---------------- #

recognizer = cv2.face.LBPHFaceRecognizer_create()
model_loaded = False

if os.path.exists(TRAINER_PATH) and os.path.getsize(TRAINER_PATH) > 0:
    recognizer.read(TRAINER_PATH)
    model_loaded = True

# ---------------- USERS ---------------- #

if os.path.exists(USERS_PATH):
    with open(USERS_PATH, "r") as f:
        try:
            users = json.load(f)
        except json.JSONDecodeError:
            users = {}
else:
    users = {}

# ---------------- FACE DETECTORS ---------------- #

face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

if face_cascade.empty():
    raise RuntimeError("Failed to load Haar cascade")

if not os.path.exists(PREDICTOR_PATH):
    raise RuntimeError(
        f"Dlib model not found at:\n{PREDICTOR_PATH}"
    )

detector = dlib.get_frontal_face_detector()
import os
print("File exists:", os.path.exists(PREDICTOR_PATH))
print("Path:", PREDICTOR_PATH)
predictor = dlib.shape_predictor(PREDICTOR_PATH)

# ---------------- GESTURES ---------------- #

GESTURES = ["Blink", "Look Left", "Look Right"]

# ---------------- EYE UTILS ---------------- #

def eye_aspect_ratio(eye):
    A = np.linalg.norm(eye[1] - eye[5])
    B = np.linalg.norm(eye[2] - eye[4])
    C = np.linalg.norm(eye[0] - eye[3])
    return (A + B) / (2.0 * C)


def get_iris_position(eye_region):
    gray = cv2.cvtColor(eye_region, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 50, 255, cv2.THRESH_BINARY_INV)

    contours, _ = cv2.findContours(
        thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )

    if contours:
        c = max(contours, key=cv2.contourArea)
        M = cv2.moments(c)
        if M["m00"] != 0:
            return int(M["m10"]/M["m00"]), int(M["m01"]/M["m00"])

    return None, None


def detect_eye_movement(landmarks, gesture, frame):

    left_eye = np.array(
        [(landmarks.part(i).x, landmarks.part(i).y) for i in range(36, 42)]
    )
    right_eye = np.array(
        [(landmarks.part(i).x, landmarks.part(i).y) for i in range(42, 48)]
    )

    x1, y1, w1, h1 = cv2.boundingRect(left_eye)
    x2, y2, w2, h2 = cv2.boundingRect(right_eye)

    left_roi = frame[y1:y1+h1, x1:x1+w1]
    right_roi = frame[y2:y2+h2, x2:x2+w2]

    lx, ly = get_iris_position(left_roi)
    rx, ry = get_iris_position(right_roi)

    if lx is None or rx is None:
        return False

    left_shift = lx / w1
    right_shift = rx / w2

    ear = (eye_aspect_ratio(left_eye) + eye_aspect_ratio(right_eye)) / 2

    if gesture == "Blink":
        return ear < 0.2

    if gesture == "Look Left":
        return left_shift > 0.7 and right_shift > 0.7

    if gesture == "Look Right":
        return left_shift < 0.3 and right_shift < 0.3

    return False


# ---------------- MAIN FUNCTION ---------------- #

def recognize_face(user_id=None):

    global model_loaded

    if not model_loaded:
        return "failure", None

    cap = cv2.VideoCapture(0)
    start_time = time.time()

    requested_gesture = random.choice(GESTURES)
    correct_gesture = False
    recognized = False
    username = None

    while time.time() - start_time < 6:

        ret, frame = cap.read()
        if not ret:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = detector(gray)

        for face in faces:
            landmarks = predictor(gray, face)

            x, y, w, h = face.left(), face.top(), face.width(), face.height()
            face_roi = gray[y:y+h, x:x+w]

            id_, confidence = recognizer.predict(face_roi)

            if confidence < 80:
                recognized = True
                username = users.get(str(id_), {}).get("name", f"User {id_}")

                if detect_eye_movement(
                    landmarks, requested_gesture, frame
                ):
                    correct_gesture = True

        cv2.putText(
            frame,
            f"Do: {requested_gesture}",
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (255, 255, 0),
            2
        )

        cv2.imshow("Face Authentication", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()

    if recognized:
        return ("success", username) if correct_gesture else ("idle", username)

    return "failure", None

