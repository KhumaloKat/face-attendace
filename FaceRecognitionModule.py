"""
Face-Recognition module 
Finds and recognises

"""

import cv2
import numpy as np 
import face_recognition
import os


def findEncodings(path):
    #images = []
    classNames = []
    encodeList = []

    myList = os.listdir(path)
    # print(myList)
    for cl in myList:
        curImg = cv2.imread(f'{path}/{cl}')
        classNames.append(os.path.splitext(cl)[0])
        curImg = cv2.cvtColor(curImg,cv2.COLOR_BGR2RGB)
        encode = face_recognition.face_encodings(curImg)[0]
        encodeList.append(encode)

    print("Encoding Completed")

    return encodeList, classNames

def recognizeFaces(img, encodeList, classNames, scaleFactor=0.25):

    imgFaces = img.copy()
    imgS = cv2.resize(img,(0,0), None, scaleFactor, scaleFactor)
    imgS = cv2.cvtColor(imgS,cv2.COLOR_BGR2RGB)
    facesCurImg = face_recognition.face_locations(imgS)
    encodeCurImage = face_recognition.face_encodings(imgS,facesCurImg)

    names = []

    for encodeFace, faceLoc in zip(encodeCurImage, facesCurImg):
        result =  face_recognition.compare_faces(encodeList,encodeFace)
        faceDis = face_recognition.face_distance(encodeList,encodeFace)
        matchIndex = np.argmin(faceDis)

        if result[matchIndex]:
            name = classNames[matchIndex].upper()
            
            color = (0,255,0)
        else: 
            color = (0,0,255)
            name= 'unknown'    
        names.append(name)
        y1, x2, y2, x1 = faceLoc
        y1, x2, y2, x1 = int(y1/scaleFactor), int(x2/scaleFactor), int(y2/scaleFactor), int(x1/scaleFactor) 
        cv2.rectangle(imgFaces,(x1,y1),(x2,y2),color,2)
        cv2.rectangle(imgFaces,(x1-40,y2-35),(x2+40,y2),color,cv2.FILLED)
        cv2.putText(imgFaces, name, (x1-40, y2-6), cv2.FONT_HERSHEY_COMPLEX,1,(255,255,255),2)

    return imgFaces, names



def main():
    encodeList, classNames = findEncodings("ImagesAttendance")
    # print(classNames)

    frameWidth = 640
    frameHeight= 480

    cap = cv2.VideoCapture(0)


    while True:
        _ , img = cap.read()
        img = cv2.resize(img, (frameWidth, frameHeight))

        imgFaces, names = recognizeFaces(img, encodeList, classNames)
        cv2.imshow("Image",imgFaces)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

if __name__ == "__main__":
    main()