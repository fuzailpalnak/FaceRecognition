import cv2

from dlib_utility import get_front_face_detector


class FaceDetector:
    def __init__(
        self,
        scale_factor=4,
    ):

        self.scale_factor = scale_factor
        self.detector = get_front_face_detector()

    def detect(self, frame):
        return frame, self.detector(frame)

    def rect_to_css(self, rect, frame, prediction_name_list):

        for i, face in enumerate(rect):
            """if want to convert dlib rectangle  and view the result"""

            """face_image = frame[face.top():face.bottom(), face.left():face.right()]
            pil_image = Image.fromarray(face_image)
            pil_image.show()
            pil_image.save("face.png")"""
            self.bbox(face, frame, prediction_name_list[i])

    @staticmethod
    def bbox(face, frame, prediction="Unknown"):

        """if want to convert dlib rectangle  and view the result"""

        x = face.left()
        y = face.top()
        w = face.right() - x
        h = face.bottom() - y
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.putText(
            frame,
            prediction,
            (x - 10, y - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 255, 0),
            2,
        )
        return x, y, w, h
