import cv2
import sys

video = cv2.VideoCapture(0)


Fpath = 'haarcascade_frontalface_default.xml'
Epath = 'haarcascade_eye.xml'

a = 1

while True:
    a = a+1
    check, frame = video.read()
    print(frame)
    face_cascade = cv2.CascadeClassifier(Fpath)
    eye_cascade = cv2.CascadeClassifier(Epath)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    face = face_cascade.detectMultiScale(gray, scaleFactor = 1.005, minNeighbors = 5, minSize = (5, 5), flags = cv2.CASCADE_SCALE_IMAGE)
    eye =  eye_cascade.detectMultiScale(gray, scaleFactor = 1.005, minNeighbors = 10, minSize = (5, 5), flags = cv2.CASCADE_SCALE_IMAGE)

    for x, y, w, h in face:
        cv2.rectangle(gray, (x, y), (x+w, y+h), (255, 0, 0), 3)

    for ex, ey, ew, eh in eye:
        cv2.rectangle(gray, (ex, ey), (ex+ew, ey+eh), (0, 0, 255), 3)

    cv2.imshow('Captured', gray)

    key = cv2.waitKey(2)
                      
    if key == ord('q'):
        break

video.release()

cv2.destroyAllWindows()





