import cv2

#img = cv2.imread('lena.png')

vcap = cv2.VideoCapture(1)
vcap.set(3, 1280)
vcap.set(4, 720)
vcap.set(10, 70)

#thres = 0.5, you can use this variable if you wanna init threshold dynamically

classNames = []
classFile = "coco.txt"
with open(classFile, 'rt') as cf:
    classNames = cf.read().rstrip('\n').split('\n')

config = "ssd_mobilenet_v3_large_coco_2020_01_14.pbtxt"
weight = "frozen_inference_graph.pb"

net = cv2.dnn_DetectionModel(weight, config)

net.setInputSize(320, 320)
net.setInputScale(1.0/127.5)
net.setInputMean((127.5, 127.5, 127.5))
net.setInputSwapRB(True)

while True:
    success, img = vcap.read()
    classIds, confs, bbox = net.detect(img, confThreshold = 0.5) # could change to variable thres instead of number value
    print(classIds, bbox)

    if len(classIds) != 0:
        for classId, confidence, box in zip(classIds.flatten(), confs.flatten(), bbox):
            cv2.rectangle(img, box, color=(0, 255, 0), thickness = 3)
            # Code use to detect & show the object class
            cv2.putText(img, classNames[classId - 1].upper(), (box[0], box[1]-10),
                        cv2.FONT_HERSHEY_COMPLEX, 1, (255, 0, 0), 2)
            # Code use to detect & show the confidence accuracy
            cv2.putText(img, str(round(confidence*100,2)), (box[0]+200, box[1]-10),
                        cv2.FONT_HERSHEY_COMPLEX, 1, (255, 0, 0), 2)                                                  
    cv2.imshow("Output", img)
    cv2.waitKey(1)
