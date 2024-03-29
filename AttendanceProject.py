import cv2
import numpy as np
import face_recognition
# untuk mengakses file
import os 
# untuk waktu dan date
from datetime import datetime


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


# fungsi absensi menggunakan nama dan waktu
def markAttendance(name):
    with open('Attendance.csv', 'r+') as f: # membuka file csv
        #agar orang yang sudah datang tidak mengulangi absen
        myDataList = f.readlines()
        nameList = []
        for line in myDataList:
            entry = line.split(',')
            nameList.append(entry[0])
        if name not in nameList:
            now = datetime.now() # waktu sekarang
            dtString = now.strftime('%H:%M:%S') # format waktu
            f.writelines(f'\n{name},{dtString}') # menulis nama dan waktu
            
        


#membuat list yang sudah di encoding
encodeListKnown = findEncodings(images)
print(len(encodeListKnown))
print('Encoding Complete')

#mengambil gambar yang akan di cocokan encodingnya dari webcam
cap = cv2.VideoCapture('http://192.168.0.107:4747/video')

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
        print(faceDis) #menampilkan jarak antara wajah
        matchIndex = np.argmin(faceDis) #mencari jarak terkecil agar bisa di cocokan dengan nama file
        
        y1, x2, y2, x1 = faceLoc
        x1, y1, x2, y2 = x1*4, y1*4, x2*4, y2*4

        # membuat kotak pada wajah dan menulis nama file
        if matches[matchIndex]:
            name = classNames[matchIndex].upper()
            print(name) #menampilkan nama file
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2) # membuat kotak pada wajah
            cv2.rectangle(img, (x1, y2-35), (x2, y2), (0, 255, 0), cv2.FILLED) # membuat kotak pada nama file
            cv2.putText(img, name, (x1+6, y2-6), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 255, 255), 2) # menulis nama file
            #memakai fungsi absensi
            markAttendance(name)
        else:
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2) # membuat kotak pada wajah
            cv2.rectangle(img, (x1, y2-35), (x2, y2), (0, 255, 0), cv2.FILLED) # membuat kotak pada nama file
            unknown = cv2.putText(img, "UNKNOWN", (x1+6, y2-6), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 255, 255), 2) # menulis "UNKNOWN"
            print(unknown) #menampilkan "UNKNOWN"
        
        cv2.imshow('Webcam', img) # menampilkan gambar
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break




