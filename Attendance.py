import cv2
import numpy as np
import face_recognition
import os
from datetime import datetime


# This is the second phase of my code lets go!!!!!!!!

# =========================================CODE BEGINS HERE======================================================= #

path = 'imagesAttendance'
images = []
classNames = []
mylist = os.listdir(path)
print(mylist)
for cl in mylist:
    curImg = cv2.imread(f'{path}/{cl}')
    images.append(curImg)
    classNames.append(os.path.splitext(cl)[0])
print(classNames)

def findEncodings(images):
    encodelist = []
    for img in images:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        encode = face_recognition.face_encodings(img)[0]
        encodelist.append(encode)
    return encodelist

def markAttendance(name):
    with open('attendance.csv', 'r+') as f:
        mydatalist = f.readlines()
        nameList = []
        for line in mydatalist:
            entry = line.split(',')
            nameList.append(entry[0])
        if name not in nameList:
            now = datetime.now()
            dtstring = now.strftime('%H:%M:%S')
            f.writelines(f'\n{name},{dtstring}')

encodelistKnown = findEncodings(images)
print('Encoding complete')

cap = cv2.VideoCapture(0)

while True:
    success, img = cap.read()
    imgS = cv2.resize(img,(0,0),None,0.25,0.25)
    imgS = cv2.cvtColor(imgS, cv2.COLOR_BGR2RGB)

    facesCurFrame = face_recognition.face_locations(imgS)
    encodeCurFrame = face_recognition.face_encodings(imgS, facesCurFrame)

    for encodeFace, Faceloc in zip(encodeCurFrame,facesCurFrame):
        matches = face_recognition.compare_faces(encodelistKnown, encodeFace)
        faceDis = face_recognition.face_distance(encodelistKnown, encodeFace)
        print(faceDis)
        matchIndex = np.argmin(faceDis)

        if matches[matchIndex]:
           name = classNames[matchIndex].upper()
           print(name)
           y1,x2,y2,x1 = Faceloc
           y1,x2,y2,x1 = y1*4,x2*4,y2*4,x1*4
           cv2.rectangle(img, (x1,y1),(x2,y2),(0,255,0),2)
           cv2.rectangle(img,(x1,y2-35),(x2,y2),(0,255,0),cv2.FILLED)
           cv2.putText(img,name,(x1+6,y2-6),cv2.FONT_HERSHEY_COMPLEX,1,(255,255,255),2)
           markAttendance(name)

    cv2.imshow('Webcam', img)
    cv2.waitKey(1)




#imgElon = face_recognition.load_image_file('imagesbasic/elon musk.jpg')
#imgElon = cv2.cvtColor(imgElon, cv2.COLOR_BGR2RGB)
# In cv2 color is taken as RGB not BGR, that's why this line is important
#imgTest = face_recognition.load_image_file('imagesbasic/elon test.jpg')
#imgTest = cv2.cvtColor(imgTest, cv2.COLOR_BGR2RGB)