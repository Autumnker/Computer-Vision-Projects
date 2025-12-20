import cv2
import pytesseract
pytesseract.pytesseract.tesseract_cmd = r'E:\Tesseract-OCR\tesseract.exe'

# Character stroke
def character_stroke(img,config=""):
    imgH,imgW,_ = img.shape
    boxes = pytesseract.image_to_boxes(img,config=config)

    for b in boxes.splitlines():
        # print(b)
        b = b.split(' ')
        # print(b)
        x,y,w,h = int(b[1]),int(b[2]),int(b[3]),int(b[4])
        cv2.rectangle(img,(x,imgH - y),(w,imgH - h),(0,0,255),3)
        cv2.putText(img,b[0],(x - 25,imgH - y + 25),cv2.FONT_HERSHEY_COMPLEX,1,(0,255,0),3)

    cv2.imshow('character_stroke',img)
    cv2.waitKey(0)

# String stroke
def string_stroke(img,config=""):
    imgH,imgW,_ = img.shape
    datas = pytesseract.image_to_data(img,config=config)

    for index,d in enumerate(datas.splitlines()):
        if index != 0:
            d = d.split()
            # print(d)
            if len(d) == 12:
                x,y,w,h = int(d[6]),int(d[7]),int(d[8]),int(d[9])
                cv2.rectangle(img,(x,y),(w + x,h + y),(0,0,255),3)
                cv2.putText(img,d[11],(x,y),cv2.FONT_HERSHEY_COMPLEX,1,(0,255,0),3)

    cv2.imshow('string_stroke',img)
    cv2.waitKey(0)

if __name__ == '__main__':
    img_path=r'Tesseract_Text_Recognition\image\test_2025-12-20_14-11-19.png'
    img = cv2.imread(img_path)
    config = r'--oem 3 --psm 6 outputbase digits'

    # character_stroke(img)
    string_stroke(img)
    # character_stroke(img,config)
    # string_stroke(img,config)
