import cv2
import numpy as np

thres = 0.45  # Threshold to detect object
nms_threshold = 0.2
cap = cv2.VideoCapture(0)
# cap.set(3,1280)
# cap.set(4,720)
# cap.set(10,150)

classNames = []
classFile = r"./ObjectDetector/coco.names"
with open(classFile, "rt") as f:
    classNames = f.read().rstrip("\n").split("\n")

# print(classNames)
configPath = r"./ObjectDetector/ssd_mobilenet_v3_large_coco_2020_01_14.pbtxt"
weightsPath = r"./ObjectDetector/frozen_inference_graph.pb"

net = cv2.dnn_DetectionModel(weightsPath, configPath)
net.setInputSize(320, 320)
net.setInputScale(1.0 / 127.5)
net.setInputMean((127.5, 127.5, 127.5))
net.setInputSwapRB(True)

while True:
    success, img = cap.read()
    if not success:
        break

    classIds, confs, bbox = net.detect(img, confThreshold=thres)
    bbox = list(bbox)
    confs = list(np.array(confs).flatten())
    confs = list(map(float, confs))
    # print(type(confs[0]))
    # print(confs)

    indices = cv2.dnn.NMSBoxes(bbox, confs, thres, nms_threshold)
    # print(indices)

    for i in indices:
        i = i[0] if isinstance(i, (np.ndarray, list)) else i
        box = bbox[i]
        x, y, w, h = box[0], box[1], box[2], box[3]
        cv2.rectangle(img, (x, y), (x + w, h + y), color=(0, 255, 0), thickness=2)
        class_id = (
            classIds[i][0]
            if isinstance(classIds[i], (np.ndarray, list))
            else classIds[i]
        )
        class_id = int(class_id)

        if class_id < 1 or class_id > len(classNames):
            label = f"Unknown (ID:{class_id})"
            print(f"UnknowID{class_id}ʾΪ{label}")
        else:
            label = classNames[class_id - 1].upper()
        cv2.putText(
            img,
            label,
            (x + 10, y + 30),
            cv2.FONT_HERSHEY_COMPLEX,
            1,
            (0, 255, 0),
            2,
        )

    cv2.imshow("Output", img)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break
