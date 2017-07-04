import dlib
import os
import glob,random
from sklearn.svm import SVC
import numpy as np
from Create_dataset.known_people_scan import *

def face_encoding(landmark_list,image,face_encoder):
    for raw_landmark_set in landmark_list:
        return True,np.array(face_encoder.compute_face_descriptor(image, raw_landmark_set))

def get_files(known_faces):
    files = glob.glob("/home/palnak/PycharmProjects/FaceRec/celebrity/%s/*.jpg" % known_faces)
    random.shuffle(files)
    training = files[:int(len(files) * 0.8)]  # get first 80% of file list
    prediction = files[-int(len(files) * 0.2):]  # get last 20% of file list
    return training, prediction


def main():
    for root, known_faces, files in os.walk("/home/palnak/PycharmProjects/FaceRec/celebrity/", True):
        break
    #known_faces = ["amyadams", "chadsmith","islafisher","willferrell","markruffalo","robertdowney","scarletjohanson","chrisevans"]
    face_recognition_model = "params/dlib_face_recognition_resnet_model_v1.dat"
    detector = dlib.get_frontal_face_detector()
    datafile_train = "/home/palnak/PycharmProjects/FaceRec/datasets/train.pkl"
    datafile_test = "/home/palnak/PycharmProjects/FaceRec/datasets/test.pkl"
    face_encoder = dlib.face_recognition_model_v1(face_recognition_model)

    training_data, training_labels, prediction_data, prediction_labels=make_set(known_faces,detector,datafile_train,datafile_test,face_encoder)


    training_data= np.squeeze(np.array(training_data))
    print training_data

    prediction_data=np.squeeze(np.array(prediction_data))
    print prediction_data
    clf = SVC(kernel='linear', probability=True,
              tol=1e-3, verbose=True)
    clf.fit(training_data, training_labels)

    f = open("params/svm_celebrity.pkl", 'wb')
    pickle.dump(clf, f)

    print("getting accuracies %s")  # Use score() function to get accuracy

    pred_lin = clf.score(prediction_data, prediction_labels)
    print pred_lin







if __name__ == '__main__':
    main()
