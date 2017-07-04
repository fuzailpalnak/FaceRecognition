from FaceDetection import FaceDetector
import dlib,cv2,numpy as np,os,pickle
image_path="/home/palnak/PycharmProjects/ExpRec/temp1.png"
landmark_data= "params/shape_predictor_68_face_landmarks.dat",
face_recognition_model = "params/dlib_face_recognition_resnet_model_v1.dat"

faces = FaceDetector()
detector = dlib.get_frontal_face_detector()

final_prediction = []
actual_prediction = []
prediction_list=[]
average_prediction_dictionary = {}
actual_probability = []
number_of_count={}
path_to_svm="params/svm.pkl"
def _rect_to_css(face, frame,prediction):
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
    cv2.putText(frame, str(prediction), (x - 10, y - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

def get_svm_classifier(path_to_svm):
    if (os.path.exists(path_to_svm)):
        f = open(path_to_svm, 'rb')
        model = pickle.load(f)
        return model

def get_average_probability(prediction,probability,prob):
    count=1
    for i in range(0, len(prob) - 1):
        if actual_prediction[i] == prediction:
            probability = probability + prob[i]
            count = count + 1
    return probability,count


def get_probability(landmark_points,face_encoding_points,model):
    for i in range(0, len(landmark_points)):

        prediction = model.predict_proba(np.array(face_encoding_points[i]).flatten()).ravel()

        max_prob = np.argmax(prediction)

        probability = prediction[max_prob]

        actual_probability.append(probability)
        prediction = model.predict(np.array(face_encoding_points[i]).flatten())
        if prediction not in actual_prediction:

            actual_prediction.append(prediction[0])
        else:
            average_probability, count =get_average_probability(prediction, probability, actual_probability)

            actual_prediction.append(prediction[0])
            average_probability = average_probability / count
        if prediction[0] not in average_prediction_dictionary:
            average_prediction_dictionary[prediction[0]] = probability
            prediction_list.append(prediction[0])
            number_of_count[prediction[0]]=1
        else:
            for i in range(0, len(average_prediction_dictionary)):
                if prediction_list[i] == prediction:
                    average_prediction_dictionary[prediction[0]] = average_probability
            number_of_count[prediction[0]]=number_of_count[prediction[0]]+1
    print number_of_count
    return actual_probability, actual_prediction, average_prediction_dictionary,number_of_count

def get_prediction(prob,prediction_list_temp,prediction_dict,number_of_count):


    for i in range(0,len(prob)):
        if number_of_count[prediction_list_temp[i]]>1:
            if prob[i]>=prediction_dict[prediction_list_temp[i]]:
                final_prediction.append(prediction_list_temp[i])
            else:
                final_prediction.append("unknown")
        else:
            print prediction_list_temp[i]
            print prob[i]
            if prob[i]>0.4:
                final_prediction.append(prediction_list_temp[i])
            else:
                final_prediction.append("unknown")

    return  final_prediction
def display_output(prediction_list,dlib_face,image):
    for i, face in enumerate(dlib_face):
        _rect_to_css(face, image, prediction_list[i])
    cv2.imwrite("im.png", image)

def main():
    image, dlib_face = faces.detect(cv2.imread(image_path), detector)
    print "num of faces found--"+ str(len(dlib_face))


    if len(dlib_face) > 0:

        success = True
    else:
        print "NO FACES FOUND"
        exit()
    if success:
        success, landmark_points = faces.face_geometry(dlib_face, image)
        success, face_encoding_points = faces.face_encoding(landmark_points, image)
        model=get_svm_classifier(path_to_svm)
        prob,prediction_list_temp,prediction_dict,number_of_count=get_probability(landmark_points,face_encoding_points,model)
        prediction_list=get_prediction(prob,prediction_list_temp,prediction_dict,number_of_count)
        display_output(prediction_list,dlib_face,image)

if __name__ == '__main__':
    main()
