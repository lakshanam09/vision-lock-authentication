import cv2
import math
import numpy as np
from Pupil import Pupil


class Eye:

    LEFT = [36, 37, 38, 39, 40, 41]
    RIGHT = [42, 43, 44, 45, 46, 47]


    def __init__(self, frame, landmarks, side, calibration):

        self.frame = None
        self.origin = None
        self.center = None
        self.pupil = None

        self.analyze(frame, landmarks, side, calibration)


    def middle(self, p1, p2):

        x = int((p1.x + p2.x) / 2)
        y = int((p1.y + p2.y) / 2)

        return x, y


    def isolate(self, frame, landmarks, points):

        region = np.array([
            (landmarks.part(p).x, landmarks.part(p).y)
            for p in points
        ])

        mask = np.zeros(frame.shape[:2], dtype=np.uint8)

        cv2.fillPoly(mask, [region], 255)

        eye = cv2.bitwise_and(frame, frame, mask=mask)

        x, y, w, h = cv2.boundingRect(region)

        self.frame = eye[y:y+h, x:x+w]

        self.origin = (x, y)

        h, w = self.frame.shape[:2]

        self.center = (w/2, h/2)


    def blink_ratio(self, landmarks, points):

        left = landmarks.part(points[0])
        right = landmarks.part(points[3])

        top = self.middle(
            landmarks.part(points[1]),
            landmarks.part(points[2])
        )

        bottom = self.middle(
            landmarks.part(points[5]),
            landmarks.part(points[4])
        )

        width = math.hypot(
            left.x-right.x, left.y-right.y
        )

        height = math.hypot(
            top[0]-bottom[0], top[1]-bottom[1]
        )

        if height == 0:
            return 0

        return width/height


    def analyze(self, frame, landmarks, side, calibration):

        points = self.LEFT if side == 0 else self.RIGHT

        self.blinking = self.blink_ratio(landmarks, points)

        self.isolate(frame, landmarks, points)

        if not calibration.is_complete():
            calibration.evaluate(self.frame, side)

        t = calibration.threshold(side)

        self.pupil = Pupil(self.frame, t)
