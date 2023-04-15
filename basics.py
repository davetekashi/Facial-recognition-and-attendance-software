import cv2
import numpy as np
import face_recognition

# this is a code to accomplish face recognition by loading faces from a directory and then detect the face
# and then determine whose face it is
# I WILL ACHIEVE THIS BY USING CV2, FACE_RECOGNITION, NUMPY(to give each point in a picture an integer)
# IN GOD I TRUST

# ============================MY CODE BEGINS HERE===================================#

imgelon = face_recognition.load_image_file('imagesbasic/elon musk.jpg')
imgelon = cv2.cvtColor(imgelon, cv2.COLOR_BGR2RGB)
# In cv2 color is taken as RGB not BGR, that's why this line is important
imgTest = face_recognition.load_image_file('imagesbasic/elon test.jpg')
imgTest = cv2.cvtColor(imgTest, cv2.COLOR_BGR2RGB)

faceLoc = face_recognition.face_locations(imgelon)[0]
encodeElon = face_recognition.face_encodings(imgelon)[0]
cv2.rectangle(imgelon, (faceLoc[3], faceLoc[0]), (faceLoc[1], faceLoc[2]), (255, 0, 255), 2)

faceLocTest = face_recognition.face_locations(imgTest)[0]
encodeTest = face_recognition.face_encodings(imgTest)[0]
cv2.rectangle(imgTest, (faceLocTest[3], faceLocTest[0]), (faceLocTest[1], faceLocTest[2]), (255, 0, 255), 2)
# This code i just wrote is for my AI to be able to identify where the face is in the image

results = face_recognition.compare_faces([encodeElon], encodeTest)
faceDis = face_recognition.face_distance([encodeElon],encodeTest)
cv2.putText(imgTest,f'{results} {round(faceDis[0],2)}', (50,50),cv2.FONT_HERSHEY_COMPLEX,1,(0,0,255),2)
# This code above shows the difference of the pictures on the picture through a boolean value
print(results, faceDis)

cv2.imshow('brie bella', imgelon)
cv2.imshow('nikki bella', imgTest)
cv2.waitKey(0)
