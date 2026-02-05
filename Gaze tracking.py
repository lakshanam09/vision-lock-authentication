from __future__ import division
import os
import cv2
import dlib

from Eye import Eye
from Calibration import Calibration


class GazeTracking(object):
    """
    This class tracks the user's gaze.
    It provides useful information like the position of the eyes
    and pupils and allows to know if the eyes are open or closed
    """

    def __init__(self):
        self.frame = None
        self.eye_left = None
        self.eye_right = None
        self.calibration = Calibration()

        # Face detector
        self._face_detector = dlib.get_frontal_face_detector()

        # Landmark predictor
        cwd = os.path.abspath(os.path.dirname(__file__))
        model_path = os.path.join(
            cwd, "trained_models/shape_predictor_68_face_landmarks.dat"
        )
        self._predictor = dlib.shape_predictor(model_path)

    @property
    def pupils_located(self):
        """Check that the pupils have been located"""
        try:
            int(self.eye_left.pupil.x)
            int(self.eye_left.pupil.y)
            int(self.eye_right.pupil.x)
            int(self.eye_right.pupil.y)
            return True
        except Exception:
            return False

    def _analyze(self):
        """Detects the face and initializes Eye objects"""
        gray = cv2.cvtColor(self.frame, cv2.COLOR_BGR2GRAY)
        faces = self._face_detector(gray)

        try:
            landmarks = self._predictor(gray, faces[0])
            self.eye_left = Eye(gray, landmarks, 0, self.calibration)
            self.eye_right = Eye(gray, landmarks, 1, self.calibration)
        except IndexError:
            self.eye_left = None
            self.eye_right = None

    def refresh(self, frame):
        """Refreshes the frame and analyzes it"""
        self.frame = frame
        self._analyze()

    def pupil_left_coords(self):
        """Returns the coordinates of the left pupil"""
        if self.pupils_located:
            x = self.eye_left.origin[0] + self.eye_left.pupil.x
            y = self.eye_left.origin[1] + self.eye_left.pupil.y
            return (x, y)

    def pupil_right_coords(self):
        """Returns the coordinates of the right pupil"""
        if self.pupils_located:
            x = self.eye_right.origin[0] + self.eye_right.pupil.x
            y = self.eye_right.origin[1] + self.eye_right.pupil.y
            return (x, y)

    def horizontal_ratio(self):
        """
        Returns a number between 0.0 and 1.0 that indicates
        the horizontal direction of the gaze
        """
        if self.pupils_located:
            left = self.eye_left.pupil.x / (self.eye_left.center[0] * 2 - 10)
            right = self.eye_right.pupil.x / (self.eye_right.center[0] * 2 - 10)
            return (left + right) / 2

    def vertical_ratio(self):
        """
        Returns a number between 0.0 and 1.0 that indicates
        the vertical direction of the gaze
        """
        if self.pupils_located:
            left = self.eye_left.pupil.y / (self.eye_left.center[1] * 2 - 10)
            right = self.eye_right.pupil.y / (self.eye_right.center[1] * 2 - 10)
            return (left + right) / 2

    def is_right(self):
        """Returns True if looking right"""
        if self.pupils_located:
            return self.horizontal_ratio() <= 0.35

    def is_left(self):
        """Returns True if looking left"""
        if self.pupils_located:
            return self.horizontal_ratio() >= 0.65

    def is_center(self):
        """Returns True if looking center"""
        if self.pupils_located:
            return not self.is_right() and not self.is_left()

    def is_blinking(self):
        """Returns True if blinking"""
        if self.pupils_located:
            blinking_ratio = (
                self.eye_left.blinking + self.eye_right.blinking
            ) / 2
            return blinking_ratio > 3.8

    def annotated_frame(self):
        """Returns the frame with pupils marked"""
        frame = self.frame.copy()

        if self.pupils_located:
            color = (0, 255, 0)

            x_left, y_left = self.pupil_left_coords()
            x_right, y_right = self.pupil_right_coords()

            cv2.line(frame, (x_left - 5, y_left),
                     (x_left + 5, y_left), color)
            cv2.line(frame, (x_left, y_left - 5),
                     (x_left, y_left + 5), color)

            cv2.line(frame, (x_right - 5, y_right),
                     (x_right + 5, y_right), color)
            cv2.line(frame, (x_right, y_right - 5),
                     (x_right, y_right + 5), color)

        return frame

