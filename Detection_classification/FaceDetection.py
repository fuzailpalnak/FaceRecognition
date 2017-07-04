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
        self.prediction=None







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


    def bbox(self,face,frame):


        """if want to convert dlib rectangle  and view the result"""


        x = face.left()
        y = face.top()
        w = face.right() - x
        h = face.bottom() - y
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.putText(frame, self.prediction, (x-10, y-10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)


    def get_prediction(self, prob, prediction_list_temp, prediction_dict, number_of_count):

        for i in range(0, len(prob)):
            if number_of_count[prediction_list_temp[i]] > 1:
                if prob[i] >= prediction_dict[prediction_list_temp[i]]:
                    self.final_prediction.append(prediction_list_temp[i])
                else:
                    self.final_prediction.append("unknown")
            else:
                # print prediction_list_temp[i]
                # print prob[i]
                if prob[i] > 0.4:
                    self.final_prediction.append(prediction_list_temp[i])
                else:
                    self.final_prediction.append("unknown")

        return True, self.final_prediction

    def get_average_probability(self, prediction, probability, prob):
        count = 1
        for i in range(0, len(prob) - 1):
            if self.actual_prediction[i] == prediction:
                probability = probability + prob[i]
                count = count + 1
        return probability, count

    def get_probability(self, landmark_points, face_encoding_points, model):
        for i in range(0, len(landmark_points)):

            prediction = model.predict_proba(np.array(face_encoding_points[i]).flatten()).ravel()

            max_prob = np.argmax(prediction)

            probability = prediction[max_prob]

            self.actual_probability.append(probability)
            prediction = model.predict(np.array(face_encoding_points[i]).flatten())
            if prediction not in self.actual_prediction:

                self.actual_prediction.append(prediction[0])
            else:
                average_probability, count = self.get_average_probability(prediction, probability,
                                                                          self.actual_probability)

                self.actual_prediction.append(prediction[0])
                average_probability = average_probability / count
            if prediction[0] not in self.average_prediction_dictionary:
                self.average_prediction_dictionary[prediction[0]] = probability
                self.prediction_list.append(prediction[0])
                self.number_of_count[prediction[0]] = 1
            else:
                for i in range(0, len(self.average_prediction_dictionary)):
                    if self.prediction_list[i] == prediction:
                        self.average_prediction_dictionary[prediction[0]] = average_probability
                self.number_of_count[prediction[0]] = self.number_of_count[prediction[0]] + 1
        print self.number_of_count
        return self.actual_probability, self.actual_prediction, self.average_prediction_dictionary, self.number_of_count

    def Face_Rec(self,landmark_list,face_encoding_points,model):
        self.model=model
        self.final_prediction = []
        self.actual_prediction = []
        self.prediction_list = []
        self.average_prediction_dictionary = {}
        self.actual_probability = []
        self.number_of_count = {}
        prob,prediction_list_temp,prediction_dict,number_of_count=self.get_probability(landmark_list,face_encoding_points,self.model)
        success , prediction_list=self.get_prediction(prob,prediction_list_temp,prediction_dict,number_of_count)
        if success:
            return success, prediction_list


