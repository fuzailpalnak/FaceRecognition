import argparse
from scipy import misc
import os
import glob
import random
from sklearn.svm import SVC
import numpy as np
import pickle

from face_detection.detection import FaceDetector
from face_recognition.recognition import FaceRecognition

from utility import make_directory


def get_known_faces(path):
    return os.listdir(path)


def generate_data_and_labels(data_items, face_detector, face_recognizer, known_faces):
    data = list()
    labels = list()
    for item in data_items:
        image = misc.imread(item)  # open image
        image, dlib_face = face_detector.detect(image)
        success, landmark_points = face_recognizer.face_geometry(dlib_face, image)
        success, face_encoding_points = face_recognizer.face_encoding(
            landmark_points, image
        )

        if success:
            data.append(np.array(face_encoding_points).flatten())
            labels.append(known_faces)
        else:
            print("SKIPPING")
    return data, labels


def get_files(known_faces, data_folder):
    files = glob.glob(
        data_folder + "/%s/*.jpg" % known_faces
    )
    random.shuffle(files)
    training = files[: int(len(files) * 0.8)]  # get first 80% of file list
    val = files[-int(len(files) * 0.2):]  # get last 20% of file list
    return training, val


def create_model(train_data, train_labels):
    train_data = np.squeeze(np.array(train_data))
    clf = SVC(kernel="linear", probability=True, tol=1e-3, verbose=True)
    clf.fit(train_data, train_labels)

    save_dir = make_directory(os.getcwd(), "params")
    f = open(os.path.join(save_dir, "face_recognition.pkl"), "wb")
    pickle.dump(clf, f)
    return clf


def validate_data(val_data, val_labels, clf):
    val_data = np.squeeze(np.array(val_data))
    print("getting accuracies %s")  # Use score() function to get accuracy

    accuracy = clf.score(val_data, val_labels)
    print(accuracy)


def main():
    parser = argparse.ArgumentParser()
    arg = parser.add_argument
    arg("--data_set_path", type=str, help="pass dataset", required=True)
    arg(
        "--rec_model",
        type=str,
        help="pass dlib_face_recognition_resnet_model_v1.dat path",
        required=True,
    )
    arg(
        "--landmark_data",
        type=str,
        help="pass shape_predictor_68_face_landmarks.dat path",
        required=True,
    )

    args = parser.parse_args()

    root_data_folder = args.data_set_path
    face_rec_model = args.rec_model
    landmark_data = args.landmark_data

    training_data = list()
    training_labels = list()

    val_data = list()
    val_labels = list()

    face_detection = FaceDetector()
    face_recognition = FaceRecognition(
        face_encoder_model=face_rec_model, landmark_data=landmark_data
    )

    known_faces = get_known_faces(root_data_folder)

    for known_faces in known_faces:
        training, prediction = get_files(known_faces, root_data_folder)
        data, labels = generate_data_and_labels(
            training, face_detection, face_recognition, known_faces
        )
        training_data.extend(data)
        training_labels.extend(labels)

        data, labels = generate_data_and_labels(
            prediction, face_detection, face_recognition, known_faces
        )

        val_data.extend(data)
        val_labels.extend(labels)

    clf = create_model(training_data, training_labels)
    validate_data(val_data, val_labels, clf)


if __name__ == "__main__":
    main()
