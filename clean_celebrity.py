import os
from Detection_classification.FaceDetection import FaceDetector
import dlib
import cv2
import glob
faces = FaceDetector()
detector = dlib.get_frontal_face_detector()
i = 0
for root, dirs, files in os.walk("/home/palnak/PycharmProjects/FaceRec/celebrity/",False):



    celebrity=os.path.join(root)
    print "Cleaning..." +celebrity

    for celebrity1 in glob.glob(celebrity+"/*.jpg"):
        image, dlib_face = faces.detect(cv2.imread(celebrity1), detector)
        if len(dlib_face) > 1 or len(dlib_face) ==0 :
            os.remove(celebrity1)
    i = i + 1