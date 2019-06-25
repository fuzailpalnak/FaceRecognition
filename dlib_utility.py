import os
import pickle

import dlib


def get_front_face_detector():
    return dlib.get_frontal_face_detector()


def get_face_recognition_model(model_path):
    return dlib.face_recognition_model_v1(model_path)


def get_landmark_detector(landmark_data_path):
    return dlib.shape_predictor(landmark_data_path)


def get_custom_trained_model(model_path):
    f = open(model_path, 'rb')
    model = pickle.load(f)
    return model
