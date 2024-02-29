import cv2
import numpy as np
import face_recognition


# STEP 1: Load the image and convert it into RGB
# import gambar 
imgElon = face_recognition.load_image_file('ImagesBasic/Elon Musk.jpg')
# convert into RGB
imgElon = cv2.cvtColor(imgElon, cv2.COLOR_BGR2RGB)

imgTest = face_recognition.load_image_file('ImagesBasic/Elon Test.jpg')
# convert into RGB
imgTest = cv2.cvtColor(imgTest, cv2.COLOR_BGR2RGB)


# STEP 2: Find the faces and encode the faces
# menemukan wajah dan menemukan gambarnya encoding 
faceLoc = face_recognition.face_locations(imgElon)[0]
# encode wajah
encodeElon = face_recognition.face_encodings(imgElon)[0]
# membuat kotak pada wajah
cv2.rectangle(imgElon, (faceLoc[3], faceLoc[0]), (faceLoc[1], faceLoc[2]), (255,0,255), 2)

#gambar test
faceLocTest = face_recognition.face_locations(imgTest)[0]
# encode wajah test
encodeTest = face_recognition.face_encodings(imgTest)[0]
# membuat kotak pada wajah
cv2.rectangle(imgTest, (faceLocTest[3], faceLocTest[0]), (faceLocTest[1], faceLocTest[2]), (255,0,255), 2)



# STEP 3: Compare the faces
# mendapatkan pengukuran 128 dimensi dan membandingkan 
results = face_recognition.compare_faces([encodeElon], encodeTest)
# menemukan jarak antara wajah  
faceDis = face_recognition.face_distance([encodeElon], encodeTest)
print(results,faceDis)

# menempatkan angka jarak pada kotak
cv2.putText(imgTest, f'{results} {round(faceDis[0],2)}', (50,50), cv2.FONT_HERSHEY_COMPLEX, 1, (0,0,255), 2)

cv2.imshow('Elon Musk', imgElon) 

cv2.imshow('Elon Test', imgTest) 
cv2.waitKey(0)



