import cv2
import numpy as np
import face_recognition
# untuk mengakses file
import os 


path = 'ImagesAttendance'   # path untuk mengakses file
images = [] # list untuk menyimpan gambar
classNames = [] # list untuk menyimpan nama file
myList = os.listdir(path) # mengakses file pada path
print(myList)
for cl in myList:
    curImg = cv2.imread(f'{path}/{cl}') # membaca file
    images.append(curImg) # menyimpan file
    classNames.append(os.path.splitext(cl)[0]) # menyimpan nama file

print(classNames)


# fungsi menghitung encoding semua gambar 
def findEncodings(images):
    encodeList = [] # list untuk menyimpan encoding
    for img in images:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) # convert into RGB
        encode = face_recognition.face_encodings(img)[0] # encoding wajah
        encodeList.append(encode) # menyimpan encoding
    return encodeList


# fungsi absensi


#membuat list yang sudah di encoding
encodeListKnown = findEncodings(images)
print(len(encodeListKnown))
print('Encoding Complete')

#mengambil gambar yang akan di cocokan encodingnya dari webcam
cap = cv2.VideoCapture(0)

while True:
    success, img = cap.read() # membaca gambar
    imgS = cv2.resize(img, (0, 0), None, 0.25, 0.25) # mengubah ukuran gambar
    imgS = cv2.cvtColor(imgS, cv2.COLOR_BGR2RGB) # convert into RGB

    faceCurFrame = face_recognition.face_locations(imgS) # menemukan wajah
    encodesCurFrame = face_recognition.face_encodings(imgS, faceCurFrame) # encoding wajah
    
    # membandingkan wajah yang diambil dengan wajah yang sudah di encoding
    
    for encodeFace, faceLoc in zip(encodesCurFrame, faceCurFrame): #mengambil wajah yang sudah di encoding dan wajah yang sudah di temukan
        matches = face_recognition.compare_faces(encodeListKnown, encodeFace) #mendapatkan pengukuran 128 dimensi dan membandingkan
        faceDis = face_recognition.face_distance(encodeListKnown, encodeFace)  #menemukan jarak antara wajah
        print(faceDis)
        matchIndex = np.argmin(faceDis) #mencari jarak terkecil agar bisa di cocokan dengan nama file
        
        
        # membuat kotak pada wajah dan menulis nama file
        if matches[matchIndex]:
            name = classNames[matchIndex].upper()
            print(name)
            y1, x2, y2, x1 = faceLoc
            x1, y1, x2, y2 = x1*4, y1*4, x2*4, y2*4
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2) # membuat kotak pada wajah
            cv2.rectangle(img, (x1, y2-35), (x2, y2), (0, 255, 0), cv2.FILLED) # membuat kotak pada nama file
            cv2.putText(img, name, (x1+6, y2-6), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 255, 255), 2) # menulis nama file
    
    cv2.imshow('Webcam', img) # menampilkan gambar
    cv2.waitKey(1) # menunggu 1 detik




# STEP 1: Load the image and convert it into RGB
# import gambar 
imgElon = face_recognition.load_image_file('ImagesBasic/Elon Musk.jpg')
# convert into RGB
imgElon = cv2.cvtColor(imgElon, cv2.COLOR_BGR2RGB)

imgTest = face_recognition.load_image_file('ImagesBasic/Elon Test.jpg')
# convert into RGB
imgTest = cv2.cvtColor(imgTest, cv2.COLOR_BGR2RGB)

