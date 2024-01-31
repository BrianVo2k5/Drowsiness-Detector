import cv2
import numpy
from scipy.spatial import distance
import imutils
import dlib
from imutils import face_utils
from playsound import playsound

def eye_aspect_ratio(eye):
    a=distance.euclidean(eye[1],eye[5])
    b=distance.euclidean(eye[2],eye[4])
    c=distance.euclidean(eye[0],eye[3])
    dis=(a+b)/(2.0*c)
    return dis

frame_check=30
thresh=0.25
detect=dlib.get_frontal_face_detector()
eyes=dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
(lStart,lEnd)=face_utils.FACIAL_LANDMARKS_68_IDXS["left_eye"]
(rStart,rEnd)=face_utils.FACIAL_LANDMARKS_68_IDXS["right_eye"]
flag=0
cam=cv2.VideoCapture(0)
while 1:
    ret,photo=cam.read()
    photo=imutils.resize(photo,width=450)
    gray_photo=cv2.cvtColor(photo,cv2.COLOR_BGR2GRAY)
    subjects=detect(gray_photo,0)
    for subject in subjects:
        shape=eyes(gray_photo,subject)
        shape=face_utils.shape_to_np(shape)
        leftEye=shape[lStart:lEnd]
        rightEye=shape[rStart:rEnd]
        leftEar=eye_aspect_ratio(leftEye)
        rightEar=eye_aspect_ratio(rightEye)
        ear=(leftEar+rightEar)/2.0
        lh=cv2.convexHull(leftEye)
        rh=cv2.convexHull(rightEye)
        cv2.drawContours(photo,[lh],-1,(255,255,0),1)
        cv2.drawContours(photo,[rh],-1,(255,255,0),1)
        if ear < thresh:
            flag+=1
            print(flag)
            if flag > 15:
                cv2.putText(photo,"Open Your Eyes!",(10,30),cv2.FONT_HERSHEY_SIMPLEX,0.7,(255,255,0),2)
                print("Drowsiness detected")
                playsound('Alert.mp3')
        else:   
            flag=0
        cv2.imshow("photo", photo)
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
cv2.destroyAllWindows()
cap.stop()         
            

