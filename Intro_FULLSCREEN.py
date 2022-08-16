import cv2
import os
import time
from time import sleep
from tensorflow import keras
from playsound import playsound
import numpy as np
from _datetime import date ,datetime

Video_cap = cv2.VideoCapture(0)
ret, img = Video_cap.read()

Output = ['60','80','100','120']
modeltest = keras.models.load_model('D:/Data/Nhadangtocdo/model/car44.h5')
pTime = 0

# FULL_SCREEN
#cv2.namedWindow("Khoa Luan Tot Nghiep", cv2.WND_PROP_FULLSCREEN)
#cv2.setWindowProperty("Khoa Luan Tot Nghiep",cv2.WND_PROP_FULLSCREEN,cv2.WINDOW_FULLSCREEN)

# Intro
#img2 = cv2.imread("logo.jfif")
#cv2.imshow('Khoa Luan Tot Nghiep', img2)
#time.sleep(5)
#img2 = cv2.VideoCapture(0)

# Im tấm ảnh lên khung CAM
FolderPath = "car"
lst = os.listdir(FolderPath)
lst_2 = []

# FPS update time in seconds
start_time = time.time()
display_time = 0
fc = 0
FPS = 0
for i in lst:
    mau = cv2.imread(f"{FolderPath}/{i}")
    #print(f"{FolderPath}/{i}")
    lst_2.append(mau)           # 0-100 | 1-120 | 2-60 | 3-80

while True:
    ret, img = Video_cap.read()
    img1 = cv2.flip(img, 1)
    img2 = cv2.flip(img1, 1)
    img = np.asarray(img)
    img = cv2.resize(img, (100, 100))
    img = img.reshape(1, 100, 100, 3)
    fc += 1
    TIME = time.time() - start_time


    if (TIME) >= display_time:
        FPS = fc / (TIME)
        fc = 0
        start_time = time.time()

    fps_disp = "FPS: " + str(FPS)[:5]


    # Vẽ nền và khung cho Date_Time_FPS
    cv2.rectangle(img2, (100, 0), (650, 40), (137, 137, 137), -1) # (x_trái, y_trên ; x_phải,y_dưới)
    cv2.rectangle(img2, (100, 1), (650, 40), (0, 0, 0), 2)  # (x_trái, y_trên ; x_phải,y_dưới)

    # Thời gian
    #Ngày
    today = date.today()
    ngay = today.day
    thang = today.month
    nam = today.year
    cv2.putText(img2, f"{int(ngay)}" + f":{int(thang)}" + f":{int(nam)}",
                (130, 30), cv2.FONT_HERSHEY_PLAIN, 1.5, (0, 0, 0), 2)

    # Thứ
    thus = ["Thu 2", "Thu 3", "Thu 4", "Thu 5", "Thu 6", "Thu 7", "Chu Nhat"]
    thu = date.today()
    day = thu.weekday()

    # Giờ-phút-giây
    HMS = datetime.now()
    hour = HMS.hour
    minute = HMS.minute
    sec = HMS.second
    cv2.putText(img2,thus[day] +" "+ f"{int(hour)}" + f":{int(minute)}" + f":{int(sec)}",
                (270, 30), cv2.FONT_HERSHEY_PLAIN, 1.5, (0, 0, 0), 2)

    # FPS
    #cTime = time.time()
    #fps = 1/(cTime-pTime)
    #pTime=cTime
    #cv2.putText(img2,f"FPS:{int(fps)}",(520,30), cv2.FONT_HERSHEY_PLAIN,1.5,(0,0,0),2)
    #cv2.putText(img2, fps_disp, (520,30),cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    img3=cv2.putText(img2, fps_disp, (500, 30), cv2.FONT_HERSHEY_PLAIN, 1.5, (0, 0, 0), 2)

    # Vẽ nền và khung cho text
    cv2.rectangle(img2, (-30, 480), (650, 435), (137, 137, 137), -1)  # (x_trái, y_dưới ; x_phải,y_trên)
    cv2.rectangle(img2, (-30, 478), (650, 435), (0, 0, 0), 2)

    # Img trên Frame
    h, w, c = lst_2[0].shape
    #img2[0:h,0:w] = lst_2[0]

    # Dự đoán
    predictions = modeltest.predict(img)
    probabilityValue = np.amax(predictions)

    if probabilityValue > 0.7:
        if Output[np.argmax(predictions)] == '60':
            cv2.putText(img2, "Toc do gioi han - 60" + "km/h",
                        (30, 468), cv2.FONT_HERSHEY_PLAIN, 2, (0, 0, 0), 2)
            img2[0:h, 0:w] = lst_2[2]
            print("Biển 60km/h")
            #playsound('sound60.mp3', False)
            #sleep(3)
        if Output[np.argmax(predictions)] == '80':
            cv2.putText(img2, "Toc do gioi han - 80" + "km/h",
                        (30, 468), cv2.FONT_HERSHEY_PLAIN, 2, (0, 0, 0), 2)
            img2[0:h, 0:w] = lst_2[3]
            print("Biển 80km/h")
            #playsound('sound80.mp3', False)
        if Output[np.argmax(predictions)] == '100':
            cv2.putText(img2, "Toc do gioi han - 100" + "km/h",
                        (30, 468), cv2.FONT_HERSHEY_PLAIN, 2, (0, 0, 0), 2)
            img2[0:h, 0:w] = lst_2[0]
            print("Biển 100km/h")
            #playsound('sound100.mp3', False)
        if Output[np.argmax(predictions)] == '120':
            cv2.putText(img2, "Toc do gioi han - 120" + "km/h",
                        (30, 468), cv2.FONT_HERSHEY_PLAIN, 2, (0, 0, 0), 2)
            img2[0:h, 0:w] = lst_2[1]
            print("Biển 120km/h")
            #playsound('sound120.mp3', False)
    else:
        cv2.putText(img2, 'Khong phat hien bien bao toc do',
                    (30, 468), cv2.FONT_HERSHEY_PLAIN, 2, (0, 0, 0), 2)
        print("Không phát hiện biển báo tốc độ");

    cv2.imshow('Khoa Luan Tot Nghiep', img3)

    key = cv2.waitKey(1)
    if key & 0xFF == ord('q') or key == 27 or 'x' == chr(key & 255):
        break


Video_cap.release()
cv2.destroyAllWindows()
