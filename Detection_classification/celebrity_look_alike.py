from FaceDetection import FaceDetector
import dlib,cv2,numpy as np,os,pickle
image_path="/home/palnak/Downloads/neutral.jpg"
landmark_data= "params/shape_predictor_68_face_landmarks.dat",
face_recognition_model = "params/dlib_face_recognition_resnet_model_v1.dat"
import matplotlib.pyplot as plt
faces = FaceDetector()
detector = dlib.get_frontal_face_detector()

final_prediction = []
actual_prediction = []
prediction_list=[]
average_prediction_dictionary = {}
actual_probability = []
number_of_count={}
path_to_svm="params/svm.pkl"
path_to_celeb_svm="params/celeb_classifier/"
# def _rect_to_css(face, frame,prediction):
#     """if want to convert dlib rectangle  and view the result"""
#     """face_image = frame[face.top():face.bottom(), face.left():face.right()]
#     pil_image = Image.fromarray(face_image)
#     pil_image.show()
#     pil_image.save("face.png")"""
#     x = face.left()
#     y = face.top()
#     w = face.right() - x
#     h = face.bottom() - y
#     cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0),2)
#     cv2.putText(frame, str(prediction), (x - 10, y - 10),
#                 cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
def get_face(face, image):
    x = face.left()
    y = face.top()
    w = face.right()
    h = face.bottom()
    face_image = image[y-80:h+60, x-20:w+60]

    return face_image

def get_svm_classifier(path_to_svm):
    if (os.path.exists(path_to_svm)):
        f = open(path_to_svm, 'rb')
        model = pickle.load(f)
        return model



def get_prediction(face_encoding_point, model):
    prediction = model.predict(np.array(face_encoding_point).flatten())
    return prediction


# def display_output(prediction,dlib_face,image):
#     _rect_to_css(dlib_face[0], image, prediction)
#     cv2.imwrite("im.png", image)
#     return 1


def main():
    image, dlib_face = faces.detect(cv2.imread(image_path), detector)
    print "num of faces found--"+ str(len(dlib_face))
    if len(dlib_face) > 1:
        print "WARNING: More than one face found  Only considering the first face found."


    if len(dlib_face) > 0:


        success = True
    else:
        print "NO FACES FOUND"
        exit()
    if success:
        success, landmark_points = faces.face_geometry(dlib_face, image)
        success, face_encoding_points = faces.face_encoding(landmark_points, image)
        face_encoding_point=np.array(face_encoding_points)[0]

        model=get_svm_classifier(path_to_svm)
        prediction=get_prediction(face_encoding_point, model)

        celeb_model=get_svm_classifier(path_to_celeb_svm+str(prediction[0])+".pkl")
        celeb_prediction=get_prediction(face_encoding_point,celeb_model)
        celeb_image, celeb_dlib_face = faces.detect(cv2.imread(celeb_prediction[0]), detector)


        celeb_image=get_face(celeb_dlib_face[0], celeb_image)
        image=get_face(dlib_face[0],image)

        plt.title("Celebrity Look a Like")

        plt.imshow(cv2.cvtColor(celeb_image,cv2.COLOR_RGB2BGR))
        plt.show()
        # flag=display_output(prediction[0],dlib_face,image)
        #
        #
        # if flag==1:
        #     print "YOUR CELEBRITY LOOKALIKE SAVED"
        # else:
        #     print "OOPS SOMETHING WENT WRONG...."

if __name__ == '__main__':
    main()
