import cv2
import wx

from gui import BaseLayout
from face_detection.detection import FaceDetector
from face_recognition.recognition import FaceRecognition

from dlib_utility import get_custom_trained_model


class FaceRecVideo(BaseLayout):
    def __init__(self, capture, title, face_encoder_file, landmark_data, model_path):
        super().__init__(capture, title)

        self.samples = []
        self.labels = []

        self.face_detect = FaceDetector()
        self.face_rec = FaceRecognition(
            face_encoder_model=face_encoder_file, landmark_data=landmark_data
        )

        self.model = get_custom_trained_model(model_path)
        self.count = 0

    def _init_custom_layout(self):
        self.Bind(wx.EVT_CLOSE, self._on_exit)

    def init_algorithm(self):
        pass

    def _create_custom_layout(self):
        """ IF want to add custom features for frame """
        pass

    def _process_frame(self, frame):

        frame, dlib_face = self.face_detect.detect(frame)

        if len(dlib_face) > 0:

            success, landmark_points = self.face_rec.face_geometry(dlib_face, frame)
            success, face_encoding_points = self.face_rec.face_encoding(
                landmark_points, frame
            )

            if self.count % 2 == 0:
                prob, prediction_list_temp, prediction_dict, number_of_count = self.face_rec.get_probability(
                    landmark_points, face_encoding_points, self.model
                )
                success, prediction_list = self.face_rec.get_prediction(
                    prob, prediction_list_temp, prediction_dict, number_of_count
                )

                self.face_detect.rect_to_css(dlib_face, frame, prediction_list)

            self.count = self.count + 1

        else:
            print("NO FACE DETECTED")
        return frame

    def _on_snapshot(self, evt):
        """TODO add  code to save new faces
        """

    def _on_exit(self, evt):
        self.Destroy()


def main(landmark_data, face_rec_model, custom_model_path):
    capture = cv2.VideoCapture(0)
    if not (capture.isOpened()):
        capture.open()

    # capture.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    # capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    # start graphical user interface
    app = wx.App()
    layout = FaceRecVideo(capture, title="Face Detection", face_encoder_file=face_rec_model,
                          landmark_data=landmark_data, model_path=custom_model_path)
    layout.init_algorithm()
    layout.Show(True)
    app.MainLoop()


if __name__ == "__main__":
    land_mark_data = "/home/palnak/PersonalProjects/python_proj/params/shape_predictor_68_face_landmarks.dat"
    face_model = "/home/palnak/PersonalProjects/python_proj/params/dlib_face_recognition_resnet_model_v1.dat"
    model = "/home/palnak/PersonalProjects/python_proj/FaceRecognition/params/face_recognition.pkl"
    main(land_mark_data, face_model, model)
