from FaceDetection import FaceDetector
import dlib,cv2,numpy as np,os,pickle
image_path="/home/palnak/Downloads/test1.png"
landmark_data= "params/shape_predictor_68_face_landmarks.dat",
face_recognition_model = "params/dlib_face_recognition_resnet_model_v1.dat"

faces = FaceDetector()
detector = dlib.get_frontal_face_detector()
prediction_list = []


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


if (os.path.exists("params/svm.pkl")):
    f = open("params/svm.pkl", 'rb')
    model = pickle.load(f)



image,dlib_face = faces.detect(cv2.imread(image_path),detector)
print len(dlib_face)
print dlib_face

if len(dlib_face) > 0:
    # print len(self.dlib_face)

    # cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
    success = True
    #cv2.imwrite("im.png",image)
if success:
    success, landmark_points = faces.face_geometry(dlib_face, image)
print len(landmark_points)
success, face_encoding_points = faces.face_encoding(landmark_points,image)
for i in range(0, len(landmark_points)):

    # self.face_encoding_points = np.array(self.face_encoding_points).flatten()
    prediction= model.predict_proba(np.array(face_encoding_points[i]).flatten()).ravel()

    max_prob = np.argmax(prediction)
    print prediction[max_prob]
    if prediction[max_prob] > 0.6:
        prediction =model.predict(np.array(face_encoding_points[i]).flatten())
        prediction_list.append(prediction[0])
    else:
        prediction_list.append("unknown")
print prediction_list

for i,face in enumerate(dlib_face):
    _rect_to_css(face, image,prediction_list[i])
cv2.imwrite("im.png",image)
