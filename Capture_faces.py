import cv2
import os

def capture_face_images(user_id, save_path="dataset", num_images=20):

    detector = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
    cam = cv2.VideoCapture(0)

    user_folder = os.path.join(save_path, f"user_{user_id}")

    if os.path.exists(user_folder):
        print("User already registered!")
        return False

    os.makedirs(user_folder)

    count = 0

    while count < num_images:

        ret, frame = cam.read()
        if not ret:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        faces = detector.detectMultiScale(gray, 1.2, 5)

        for (x, y, w, h) in faces:

            count += 1

            file = f"{user_folder}/{count}.jpg"

            cv2.imwrite(file, gray[y:y+h, x:x+w])

            cv2.rectangle(
                frame, (x, y), (x+w, y+h), (0, 255, 0), 2
            )

        cv2.imshow("Capture Faces", frame)

        if cv2.waitKey(100) & 0xFF == ord("q"):
            break

    cam.release()
    cv2.destroyAllWindows()

    if count > 0:
        print(f"✅ {count} images saved")
        return True

    print("❌ No images captured")
    return False
