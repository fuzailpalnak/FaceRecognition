import cv2
import wx

from gui import BaseLayout
from face_detection.detection import FaceDetector


class FaceDetectVideo(BaseLayout):

    def __init__(self, capture, title):
        super().__init__(capture, title)

        self.samples = []
        self.labels = []

        self.face_detect = FaceDetector()

    def _init_custom_layout(self):
        self.Bind(wx.EVT_CLOSE, self._on_exit)

    def init_algorithm(self):
        pass

    def _create_custom_layout(self):
        """ IF want to add custom features for frame """
        pass

    def _process_frame(self, frame):

        frame, dlib_face = self.face_detect.detect(frame)

        if len(dlib_face) >0:
            for face in dlib_face:
                self.face_detect.bbox(face, frame)

        else:
            print("NO FACE DETECTED")
        return frame

    def _on_snapshot(self, evt):
        """TODO add  code to save new faces
        """

    def _on_exit(self, evt):
        self.Destroy()


def main():
    capture = cv2.VideoCapture(0)
    if not(capture.isOpened()):
        capture.open()

    # capture.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    # capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    # start graphical user interface
    app = wx.App()
    layout = FaceDetectVideo(capture, title='Face Detection')
    layout.init_algorithm()
    layout.Show(True)
    app.MainLoop()


if __name__ == '__main__':
    main()
