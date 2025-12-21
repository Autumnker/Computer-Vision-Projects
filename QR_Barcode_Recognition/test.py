import cv2
import numpy as np
from pyzbar.pyzbar import decode

cap = cv2.VideoCapture(0)
cap.set(3,640)
cap.set(4,480)

with open(r'QR_Barcode_Recognition\database.txt') as f:
    dataList = f.read().splitlines()
# print(dataList)

while True:
    status,img = cap.read()
    for barcode in decode(img):
        myData = barcode.data.decode("utf-8")
        if myData in dataList:
            color = (0,255,0)
            info = myData + ' Authorized'
        else:
            color = (0,0,255)
            info = myData + ' Un-Authorized'

        pts = np.array([barcode.polygon],np.int32)
        pts = pts.reshape((-1,1,2))
        cv2.polylines(img,[pts],True,color,5)
        pts2 = barcode.rect
        cv2.putText(img,info,(pts2[0],pts2[1]),cv2.FONT_HERSHEY_COMPLEX,0.9,color,2)

    cv2.imshow('test',img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        exit()
