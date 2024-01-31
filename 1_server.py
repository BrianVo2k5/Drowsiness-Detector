import pickle
import socket
import struct
import numpy as np
import imutils
from imutils.object_detection import non_max_suppression
import cv2
import dlib
from imutils import face_utils
from scipy.spatial import distance
# Auto get IP address WIFI
ip = socket.gethostbyname_ex(socket.gethostname())[-1][-1]
print("[INFO] Server IP: {}".format(ip))
detect=dlib.get_frontal_face_detector()
faceCascade = cv2.CascadeClassifier('./haarcascade_frontalface_alt.xml')
eyes=dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
(lStart,lEnd)=face_utils.FACIAL_LANDMARKS_68_IDXS["left_eye"]
(rStart,rEnd)=face_utils.FACIAL_LANDMARKS_68_IDXS["right_eye"]
flag=0
thresh = 0.25
HOST = ip
PORT = 8089

s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
print('Socket created')

s.bind((HOST, PORT))
print('Socket bind complete')
s.listen(10)
print('Socket now listening')

conn, addr = s.accept()
print('Connected with ' + addr[0] + ':' + str(addr[1]))
data = b'' # data is a byte string
payload_size = struct.calcsize("L")

# Load the model
def eye_aspect_ratio(eye):
    a=distance.euclidean(eye[1], eye[5])
    b=distance.euclidean(eye[2], eye[4])
    c=distance.euclidean(eye[0], eye[3])
    dis=(a+b)/(2.0*c)
    return dis
while True:
        # Retrieve message size first
    while len(data) < payload_size:
        data += conn.recv(1024)

    packed_msg_size = data[:payload_size]
    data = data[payload_size:]
    msg_size = struct.unpack("!i", packed_msg_size)[0]  # CHANGED
        
        # Retrieve all data based on message size
    while len(data) < msg_size:
        data += conn.recv(1024)
        
        # Set frame data by unpacking data and resetting data variable
    frame_data = data[:msg_size]
    data = data[msg_size:]

        # Extract frame
    frame = np.frombuffer(frame_data, dtype=np.uint8).reshape(864, 480, 4)

        # To RGB
    frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)

        # Resize
    frame = cv2.resize(frame, (300,300))

        # Show frame (only for debug) - If turn it on (uncomment/remove #), the stop and restart facial detection again can be failed on mobile.
    cv2.imshow('frame', frame)
        # Detect face
    gray_photo = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
    faces = faceCascade.detectMultiScale(gray_photo,
                    scaleFactor=1.1,
                    minNeighbors=5,
                    minSize=(30, 30),
                    flags = cv2.CASCADE_SCALE_IMAGE)
    faces = np.array([[x, y, x + w, y + h] for (x, y, w, h) in faces])
    faces = non_max_suppression(faces, probs=None, overlapThresh=0.65)
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
        cv2.drawContours(frame,[lh],-1,(255,255,0),1)
        cv2.drawContours(frame,[rh],-1,(255,255,0),1)
        if ear < thresh:
            flag +=1
            print(flag)
            if flag > 2:
                conn.sendall(str([1]).encode())
            else:
                conn.sendall(str('').encode())
        else:
            flag = 0
            print(flag)
        print(faces)
    cv2.waitKey(1)
#(except Exception as e: 
    #print(e)
    #conn.close()
    #print('Disconnected')
        # if client is disconnected, reconnect
    #conn, addr = s.accept()
    #print('Reconnected with ' + addr[0] + ':' + str(addr[1]))
    #continue
