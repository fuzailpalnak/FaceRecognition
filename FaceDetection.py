import dlib
import cv2
import numpy as np




class FaceDetector:


    def __init__(
            self,
            face_recognition_model="params/dlib_face_recognition_resnet_model_v1.dat",
            landmark_data="params/shape_predictor_68_face_landmarks.dat",
            scale_factor=4):

        self.scale_factor = scale_factor
        self.face_encoder=dlib.face_recognition_model_v1(face_recognition_model)
        self.predictor = dlib.shape_predictor(landmark_data)
        self.prediction="fuzail"





    def detect(self, frame,face_detector):

        return frame,face_detector(frame)

    def _rect_to_css(self,rect,frame):

        for face in rect:
            """if want to convert dlib rectangle  and view the result"""

            """face_image = frame[face.top():face.bottom(), face.left():face.right()]
            pil_image = Image.fromarray(face_image)
            pil_image.show()
            pil_image.save("face.png")"""
            x = face.left()
            y = face.top()
            w = face.right() - x
            h = face.bottom() - y
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(frame, str(self.prediction), (x - 10, y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

            return (x, y, w, h)

    def face_geometry(self,dlib_face,frame):

        return True,[self.predictor(frame, face_location) for face_location in dlib_face]


    def face_encoding(self,landmark_list,frame):
        return True, [np.array(self.face_encoder.compute_face_descriptor(frame, raw_landmark_set)) for
           raw_landmark_set in landmark_list]


