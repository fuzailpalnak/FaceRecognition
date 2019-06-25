import cv2

from dlib_utility import get_custom_trained_model
from face_detection.detection import FaceDetector
from face_recognition.recognition import FaceRecognition


def main(landmark_data, face_rec_model, custom_model_path):
    img_path = "/home/palnak/Desktop/test.jpg"
    face_detect = FaceDetector()
    face_rec = FaceRecognition(
        face_encoder_model=face_rec_model, landmark_data=landmark_data
    )

    model = get_custom_trained_model(custom_model_path)

    img = cv2.imread(img_path)

    frame, dlib_face = face_detect.detect(img)

    if len(dlib_face) > 0:

        success, landmark_points = face_rec.face_geometry(dlib_face, frame)
        success, face_encoding_points = face_rec.face_encoding(
            landmark_points, frame
        )

        prob, prediction_list_temp, prediction_dict, number_of_count = face_rec.get_probability(
            landmark_points, face_encoding_points, model
        )
        success, prediction_list = face_rec.get_prediction(
            prob, prediction_list_temp, prediction_dict, number_of_count
        )

        face_detect.rect_to_css(dlib_face, frame, prediction_list)
        cv2.imwrite("detect.png", frame)


if __name__ == "__main__":
    land_mark_data = "/home/palnak/PersonalProjects/python_proj/params/shape_predictor_68_face_landmarks.dat"
    face_model = "/home/palnak/PersonalProjects/python_proj/params/dlib_face_recognition_resnet_model_v1.dat"
    model = "/home/palnak/PersonalProjects/python_proj/FaceRecognition/params/face_recognition.pkl"
    main(land_mark_data, face_model, model)
