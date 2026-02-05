import cv2
from Pupil import Pupil


class Calibration:

    def __init__(self):

        self.nb_frames = 20
        self.thresholds_left = []
        self.thresholds_right = []


    def is_complete(self):

        return (
            len(self.thresholds_left) >= self.nb_frames and
            len(self.thresholds_right) >= self.nb_frames
        )


    def threshold(self, side):

        if side == 0:
            return int(sum(self.thresholds_left) / len(self.thresholds_left))

        if side == 1:
            return int(sum(self.thresholds_right) / len(self.thresholds_right))


    @staticmethod
    def iris_size(frame):

        frame = frame[5:-5, 5:-5]

        h, w = frame.shape[:2]

        total = h * w

        blacks = total - cv2.countNonZero(frame)

        return blacks / total


    @staticmethod
    def find_best_threshold(eye):

        avg = 0.48

        trials = {}

        for t in range(5, 100, 5):

            img = Pupil.image_processing(eye, t)

            trials[t] = Calibration.iris_size(img)

        best, _ = min(
            trials.items(),
            key=lambda x: abs(x[1] - avg)
        )

        return best


    def evaluate(self, eye, side):

        t = self.find_best_threshold(eye)

        if side == 0:
            self.thresholds_left.append(t)

        if side == 1:
            self.thresholds_right.append(t)
