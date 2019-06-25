import numpy as np

from dlib_utility import get_landmark_detector, get_face_recognition_model


class FaceRecognition:
    def __init__(self, landmark_data, face_encoder_model):
        self.predictor = get_landmark_detector(landmark_data)
        self.face_encoder = get_face_recognition_model(face_encoder_model)

    @staticmethod
    def get_average_probability(actual_prediction, prediction, probability, prob):
        count = 1
        for i in range(0, len(prob) - 1):
            if actual_prediction[i] == prediction:
                probability = probability + prob[i]
                count = count + 1
        return probability, count

    def get_probability(self, landmark_points, face_encoding_points, model):
        actual_prediction = []
        prediction_list = []
        average_prediction_dictionary = {}
        actual_probability = []
        number_of_count = {}
        average_probability = None

        for i in range(0, len(landmark_points)):
            encoding_points = np.array(face_encoding_points)[i:]
            prediction = model.predict_proba(encoding_points).ravel()

            max_prob = np.argmax(prediction)

            probability = prediction[max_prob]

            actual_probability.append(probability)
            prediction = model.predict(encoding_points)
            if prediction not in actual_prediction:

                actual_prediction.append(prediction[0])
            else:
                average_probability, count = self.get_average_probability(
                    actual_prediction, prediction, probability, actual_probability
                )

                actual_prediction.append(prediction[0])
                average_probability = average_probability / count
            if prediction[0] not in average_prediction_dictionary:
                average_prediction_dictionary[prediction[0]] = probability
                prediction_list.append(prediction[0])
                number_of_count[prediction[0]] = 1
            else:
                for j in range(0, len(average_prediction_dictionary)):
                    if prediction_list[j] == prediction:
                        average_prediction_dictionary[
                            prediction[0]
                        ] = average_probability
                number_of_count[prediction[0]] = number_of_count[prediction[0]] + 1
        return (
            actual_probability,
            actual_prediction,
            average_prediction_dictionary,
            number_of_count,
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
