import cv2
import math
import numpy as np

img_path = r'AngleFinder\image\test_2025-12-20_19-38-22.png'
try:
    original_img = cv2.imread(img_path)
    if original_img is None:
        raise Exception(f"Can't get image,please check image path:{img_path}")
except Exception as e:
    print(e)
    exit()

img = original_img.copy() # use backup file
pointList = []
not_printed = True

# Predefined
SHAPE_POINTS_NUMBER=3   # Unalterable
POINT_PIXEL_SIZE=3
LINE_PIXEL_SIZE=1
NUMBER_PIXEL_SIZE=0.5
NUMBER_OFFSET_PIXEL_X=-25
NUMBER_OFFSET_PIXEL_Y=25
POINT_COLOR=(255,0,0)
NUMBER_COLOR=(0,255,0)
LINE_COLOR=(0,0,255)

def mousePoints(event,x,y,flags,params):
    global img, pointList, not_printed
    if event == cv2.EVENT_LBUTTONDOWN:
        not_printed = True
        cv2.circle(img,(x,y),POINT_PIXEL_SIZE,POINT_COLOR,cv2.FILLED)
        pointList.append([x,y])
        print("-----\n",pointList,"\n-----\n")
    elif event == cv2.EVENT_RBUTTONDOWN:
        pointList = []
        img = original_img.copy()
        print("clear all")

def getAngle(pt1,pt2,pt3):
    v1 = pt1 - pt2
    v2 = pt3 - pt2

    dot_product = np.dot(v1,v2)
    norm_v1 = np.linalg.norm(v1)
    norm_v2 = np.linalg.norm(v2)

    if norm_v1 == 0 or norm_v2 == 0:
        return 0.0
    
    cos_theta = np.clip(dot_product / (norm_v1 * norm_v2), -1.0, 1.0)
    theta = math.acos(cos_theta)
    angle = math.degrees(theta)

    return angle

# register callback function 'setMouseCallback'
cv2.imshow('test',img)
cv2.setMouseCallback('test',mousePoints)

# main loop
while True:
    if not_printed and len(pointList) != 0 and len(pointList) % SHAPE_POINTS_NUMBER == 0:
        not_printed = False
        pt1,pt2,pt3 = pointList[-SHAPE_POINTS_NUMBER:]
        cv2.line(img,pt1,pt2,LINE_COLOR,LINE_PIXEL_SIZE)
        cv2.line(img,pt2,pt3,LINE_COLOR,LINE_PIXEL_SIZE)
        pt1 = np.array(pt1)
        pt2 = np.array(pt2)
        pt3 = np.array(pt3)
        angle = getAngle(pt1,pt2,pt3)
        cv2.putText(img,str(f"{angle:.2f}"),
                    pt2 + (NUMBER_OFFSET_PIXEL_X,NUMBER_OFFSET_PIXEL_Y),cv2.FONT_HERSHEY_COMPLEX,NUMBER_PIXEL_SIZE,NUMBER_COLOR)
        print(f"angle is : {angle:.2f}°")

    cv2.imshow('test',img)
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break

# release window
cv2.destroyAllWindows()
