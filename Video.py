import os
import torch
from Model import Resnet34
import torch.nn.functional as F
import numpy as np
import cv2
import sys


# load the model
model = Resnet34()
model.load_state_dict(torch.load("MASK_DET_model.pt", map_location=torch.device('cpu')))
labels = ['Mask', 'No Mask']

if torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device("cpu")

model = model.to(device)

# To capture video from webcam.
cap = cv2.VideoCapture(0)
proto = os.path.join("D:/Mask Detection", "deploy.prototxt.txt")

res10 = os.path.join("D:/Mask Detection", "res10_300x300_ssd_iter_140000.caffemodel")

network = cv2.dnn.readNetFromCaffe(proto, res10)

while True:
    # Read the frame
    _, img = cap.read()



# img = cv2.imread(os.path.join("D:/Mask Detection", "omar.jpg"))
    (height, width) = img.shape[:2]
    blob = cv2.dnn.blobFromImage(cv2.resize(img, (50, 50)), 1.0,
                             (50, 50), (104.0, 177.0, 123.0))
    network.setInput(blob)

    detections = network.forward()
    for i in range(0, detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        if confidence > 0.5:
            box = detections[0, 0, i, 3:7] * np.array([width, height, width, height])
            (startX, startY, endX, endY) = box.astype("int")
        inp = torch.tensor((np.reshape(blob, (-1, 50, 50))) / 255, dtype=torch.float32)
        with torch.no_grad():
            model.eval()
            output = model(torch.tensor(blob).squeeze(1))
            print(output)
            _, predicted = torch.max(output.data, 1)
            print(predicted)

    cv2.rectangle(img, (startX, startY), (endX, endY), (255, 0, 0), 1)
    cv2.putText(img, labels[predicted], (endX, endY - 10), cv2.FONT_HERSHEY_COMPLEX, 0.8, (255, 255, 255), 2)

#
#     # cv2.imshow("Detection", img)
#     # cv2.waitKey(0)
    cv2.imshow('img', img)
    k = cv2.waitKey(30) & 0xff
    if k == ord('q'):
        break
cap.release()


def detect(self, image):
    """ detect faces in image
    """
    net = self.classifier
    height, width = image.shape[:2]
    blob = blobFromImage(resize(image, (300, 300)), 1.0,
                         (300, 300), (104.0, 177.0, 123.0))
    network.setInput(blob)
    detections = network.forward()
    faces = []
    for i in range(0, detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        if confidence < self.confidenceThreshold:
            continue
        box = detections[0, 0, i, 3:7] * np.array([width, height, width, height])
        startX, startY, endX, endY = box.astype("int")
        faces.append(np.array([startX, startY, endX - startX, endY - startY]))
    return faces