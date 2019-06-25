import numpy as np

from dlib_utility import get_landmark_detector, get_face_recognition_model


class FaceRecognition:
    def __init__(self, landmark_data, face_encoder_model):
        self.predictor = get_landmark_detector(landmark_data)
        self.face_encoder = get_face_recognition_model(face_encoder_model)

        self.actual_prediction = []
        self.prediction_list = []
        self.average_prediction_dictionary = {}
        self.actual_probability = []
        self.number_of_count = {}
        self.average_probability = None

    @staticmethod
    def get_average_probability(actual_prediction, prediction, probability, prob):
        count = 1
        for i in range(0, len(prob) - 1):
            if actual_prediction[i] == prediction:
                probability = probability + prob[i]
                count = count + 1
        return probability, count

    def get_probability(self, landmark_points, face_encoding_points, model):

        prediction_probability = model.predict_proba(face_encoding_points)
        prediction = model.predict(face_encoding_points)
        for i in range(0, len(landmark_points)):
            max_prob = np.argmax(prediction_probability[i])

            probability = prediction_probability[i][max_prob]

            self.actual_probability.append(probability)
            if prediction[i] not in self.actual_prediction:

                self.actual_prediction.append(prediction[i])
            else:
                self.average_probability, count = self.get_average_probability(
                    self.actual_prediction, prediction[i], probability, self.actual_probability
                )

                self.actual_prediction.append(prediction[i])
                self.average_probability = self.average_probability / count
            if prediction[i] not in self.average_prediction_dictionary:
                self.average_prediction_dictionary[prediction[i]] = probability
                self.prediction_list.append(prediction[i])
                self.number_of_count[prediction[i]] = 1
            else:
                for j in range(0, len(self.average_prediction_dictionary)):
                    if self.prediction_list[j] == prediction[i]:
                        self.average_prediction_dictionary[
                            prediction[i]
                        ] = self.average_probability
                self.number_of_count[prediction[i]] = self.number_of_count[prediction[i]] + 1
        return (
            self.actual_probability,
            self.actual_prediction,
            self.average_prediction_dictionary,
            self.number_of_count,
        )

    @staticmethod
    def get_prediction(
        actual_probability,
        actual_prediction,
        average_prediction_dictionary,
        number_of_count,
    ):
        final_prediction = []

        for i in range(0, len(actual_probability)):
            if number_of_count[actual_prediction[i]] > 1:
                if (
                    actual_probability[i]
                    >= average_prediction_dictionary[actual_prediction[i]]
                ):
                    final_prediction.append(actual_prediction[i])
                else:
                    final_prediction.append("unknown")
            else:
                if actual_probability[i] > 0.4:
                    final_prediction.append(actual_prediction[i])
                else:
                    final_prediction.append("unknown")

        return True, final_prediction

    def face_geometry(self, dlib_face, frame):
        return (
            True,
            [self.predictor(frame, face_location) for face_location in dlib_face],
        )

    def face_encoding(self, landmark_list, frame):
        return (
            True,
            [
                np.array(
                    self.face_encoder.compute_face_descriptor(frame, raw_landmark_set)
                )
                for raw_landmark_set in landmark_list
            ],
        )
