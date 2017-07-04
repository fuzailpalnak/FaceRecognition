from train import get_files,face_encoding
import scipy.misc
from FaceDetection import FaceDetector
import numpy as np
from sklearn.svm import SVC
import cPickle as pickle

def create_celeb_classifier(celebrity_data,celebrity_labels,known_faces):
    print "Creating celebrity classifier.."
    celebrity_data = np.squeeze(np.array(celebrity_data))
    if len(celebrity_labels)>1:

        clf = SVC(kernel='linear', probability=True,
                  tol=1e-3, verbose=True)
        clf.fit(celebrity_data, celebrity_labels)

        f = open("params/celeb_classifier/"+known_faces+".pkl", 'wb')
        pickle.dump(clf, f)


def make_set(known_faces,detector,datafile_train,datafile_test,face_encoder):
    faces = FaceDetector("params/dlib_face_recognition_resnet_model_v1.dat", "params/shape_predictor_68_face_landmarks.dat")

    training_data = []
    celebrity_data = []
    celebrity_labels = []
    training_labels = []
    prediction_data = []
    prediction_labels = []
    print known_faces
    for known_faces in known_faces:
        training, prediction = get_files(known_faces)

        for item in training:
            print item


            image = scipy.misc.imread(item)  # open image
            image, dlib_face = faces.detect(image, detector)
            success, landmark_points = faces.face_geometry(dlib_face, image)
            success, face_encoding_points = face_encoding(landmark_points, image,face_encoder)

            if success:
                training_data.append(face_encoding_points.flatten())
                celebrity_data.append(face_encoding_points.flatten())
                celebrity_labels.append(item)
                training_labels.append(known_faces)
            else:
                print "ERROR 404!!"




        print "Creating validation set..."
        f = open(datafile_train, 'wb')
        pickle.dump(training_data, f)
        pickle.dump(training_labels, f)
        f.close()


        for item in prediction:
            print item

            image = scipy.misc.imread(item)  # open image
            image, dlib_face = faces.detect(image, detector)
            success, landmark_points = faces.face_geometry(dlib_face, image)
            # if len(landmark_points) < 1:
            #     continue
            success, face_encoding_points = face_encoding(landmark_points, image,face_encoder)
            if success:
                prediction_data.append(face_encoding_points.flatten())
                prediction_labels.append(known_faces)
                celebrity_data.append(face_encoding_points.flatten())
                celebrity_labels.append(item)
            else:
                print "ERROR 404!!!"

        f = open(datafile_test, 'wb')
        pickle.dump(prediction_data, f)
        pickle.dump(prediction_labels, f)
        f.close()

        create_celeb_classifier(celebrity_data,celebrity_labels,known_faces)
        celebrity_data = []
        celebrity_labels = []

    return training_data, training_labels, prediction_data, prediction_labels