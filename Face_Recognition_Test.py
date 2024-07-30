import cv2 
import face_recognition

frameWidth = 640
frameHeight = 480

### Import Images

img = face_recognition.load_image_file('ImagesAttendance/Katleho Khumalo.jpeg')
img2 = face_recognition.load_image_file('ImagesAttendance/Kat Khumalo.JPG')

# 
# Convert BGR to RGB

img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
img2 = cv2.cvtColor(img2,cv2.COLOR_BGR2RGB)

# Find  locations of faces

faceLoc1 = face_recognition.face_locations(img)[0]
faceLoc2 = face_recognition.face_locations(img2)[0]
# print(faceLoc1)

# Draw bounding box around faces found 

cv2.rectangle(img,(faceLoc1[3],faceLoc1[0]),(faceLoc1[1],faceLoc1[2]),(0,255,0),2)
cv2.rectangle(img2,(faceLoc2[3],faceLoc2[0]),(faceLoc2[1],faceLoc2[2]),(0,255,0),2)

# Find Encodings of images 

encode1 = face_recognition.face_encodings(img)[0]
encode2 = face_recognition.face_encodings(img2)[0]

# Compare the encodings of faces

res =  face_recognition.compare_faces([encode1],encode2)
# cv2.putText(img2,f'{res} ',(70,70),cv2.FONT_HERSHEY_COMPLEX, 3,(255,0,0),3)

faceDis = face_recognition.face_distance([encode1],encode2)
cv2.putText(img2,f'{res} : {round(faceDis[0],2)}',(70,70),cv2.FONT_HERSHEY_COMPLEX, 3,(255,0,0),3)

img = cv2.resize(img, (frameWidth, frameHeight))
img2 = cv2.resize(img2, (frameWidth, frameHeight))

cv2.imshow("Image",img)
cv2.imshow("Image 2",img2)
cv2.waitKey(0)

