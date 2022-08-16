import numpy as np #Khai báo thư viện numpy
import cv2 #Khai báo thư viện OpenCV
from tensorflow import keras #Khai báo thư viện Keras
from playsound import playsound
import os

Output = ['60','80','100','120','No'] #Biến Output chưa 2 giá trị ngõ ra sau khi nhận dạng
modeltest = keras.models.load_model('C:/Users/HuuTrien/PycharmProjects/KLTN/Model/Car32.h5')
img = cv2.imread('C:/Users/HuuTrien/PycharmProjects/KLTN/data/602.png') #lệnh đọc ảnh trong OpenCV
img1 = img
img = np.asarray(img)
img = cv2.resize(img, (100, 100)) #Chỉnh sửa kích thước ảnh
img = img.reshape(1, 100, 100, 3)
# PREDICT IMAGE
predictions = modeltest.predict(img)  #lệnh test hình ảnh trong Keras
probabilityValue = np.amax(predictions) #Truyền giá trị đoán về biến
#print(probabilityValue)
#print('Chúc bạn có một ngày mới tốt lành');
if probabilityValue > 0.7:
    print('Tỉ lệ dự đoán: ' + str(round(probabilityValue * 100, 2)) + '%')
    #print(Output[np.argmax(predictions)])
    cv2.imshow('img', img1) #Hiển thị hình ảnh lên cửa sổ img
    #if Output[np.argmax(predictions)] == '50':
        #print("Biển 50km/h")
        #playsound('sound50.mp3')
    if Output[np.argmax(predictions)] == '60':
        print("Biển 60km/h")
        #playsound('sound60.mp3')
    #if Output[np.argmax(predictions)] == '70':
        #print("Biển 70km/h")
        #playsound('sound70.mp3')
    if Output[np.argmax(predictions)] == '80':
        print("Biển 80km/h")
        #playsound('sound80.mp3')
    if Output[np.argmax(predictions)] == '100':
        print("Biển 100km/h")
        #playsound('sound100.mp3')
    if Output[np.argmax(predictions)] == '120':
        print("Biển 120km/h")
        #playsound('sound120.mp3') # Tự ghi âm
else:
    cv2.imshow('img', img1)
    print('Không phát hiện biển báo');


print("Final");
cv2.waitKey(0)
