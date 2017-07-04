import cv2
import numpy as np
import dlib
import time
import wx
from os import path
import cPickle as pickle

from Detection_classification.FaceDetection import FaceDetector
from gui import BaseLayout
import os




class FaceLayout(BaseLayout):


    def _init_custom_layout(self):

        self.samples = []
        self.labels = []


        self.Bind(wx.EVT_CLOSE, self._on_exit)

    def init_algorithm(
            self,
            save_training_file='datasets/train.pkl',
            load_svm='params/svm.pkl',

            landmark_data="params/shape_predictor_68_face_landmarks.dat",
            face_recognition_model="params/dlib_face_recognition_resnet_model_v1.dat"

    ):
        self.data_file = save_training_file
        self.faces = FaceDetector(face_recognition_model,landmark_data)
        self.detector = dlib.get_frontal_face_detector()
        # ----------Initialize the FaceDetector Constructor-----------------

        self.landmark_points = None
        self.dlib_rect=None
        self.dlib_face = None
        self.face_encoding_points=None
        self.count=0
        if (os.path.exists("params/svm.pkl")):
            f = open("params/svm.pkl", 'rb')
            self.model = pickle.load(f)
        else:
            print "MODEL NOT TRAINED "
            SystemExit(0)
        """
        TODO load existing training data from train.pkl file and load existing SVM model
        """

    def _create_custom_layout(self):
        """ IF want to add custom features for frame """


    def _process_frame(self, frame):


        # -------------------------- detect face----------------------------------------

        self.frame,self.dlib_face = self.faces.detect(frame,self.detector)
        #print self.dlib_face

        if len(self.dlib_face) >0:
            #print len(self.dlib_face)
            for face in self.dlib_face:
                self.faces.bbox(face,self.frame)
            #self.faces._rect_to_css(self.dlib_face,self.frame)
            #cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            success=True
            """ #if want to convert dlib rectangle to normal format

            self.dlib_rect = self.faces._rect_to_css(self.dlib_face,self.frame)
            print self.dlib_rect

            """

        else:
            success=False
            print "NO FACE DETECTED"

        if success :
            success, self.landmark_points= self.faces.face_geometry(self.dlib_face,self.frame)


            if success:
                #print self.landmark_points
                if (self.count%5==0):
                    success, self.face_encoding_points = self.faces.face_encoding(self.landmark_points,
                                                                                  self.frame)

                    success, prediction_list = self.faces.Face_Rec(self.landmark_points,self.face_encoding_points,
                                                                                                        self.model)


                    #print np.array(self.face_encoding_points).shape
                    if success:


                        for i, face in enumerate(self.dlib_face):


                            self.faces.prediction=prediction_list[i]









            self.count = self.count + 1
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
    layout = FaceLayout(capture, title='Face Recognition')
    layout.init_algorithm()
    layout.Show(True)
    app.MainLoop()


if __name__ == '__main__':
    main()
