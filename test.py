import cv2
from cvzone.HandTrackingModule import HandDetector
from cvzone.ClassificationModule import Classifier
import numpy as np
import math

cap = cv2.VideoCapture(0)
detector = HandDetector(maxHands=1)
classifier = Classifier("Revamped_Model/keras_model.h5", "Revamped_Model/labels.txt")
offset = 20
imgSize = 300
counter = 0

labels = ["Hello", "I love you", "No", "Okay", "Please", "Thank you", "Yes"]

while True:
    success, img = cap.read()
    if not success:
        continue
        
    imgOutput = img.copy()
    hands, img = detector.findHands(img)
    
    if hands:
        hand = hands[0]
        x, y, w, h = hand['bbox']
        

        imgHeight, imgWidth = img.shape[:2]
        x1 = max(0, x - offset)
        y1 = max(0, y - offset)
        x2 = min(imgWidth, x + w + offset)
        y2 = min(imgHeight, y + h + offset)
        
        imgCrop = img[y1:y2, x1:x2]
        
      
        if imgCrop.shape[0] == 0 or imgCrop.shape[1] == 0:
            continue
            
        imgWhite = np.ones((imgSize, imgSize, 3), np.uint8) * 255
        aspectRatio = h / w

        try:
            if aspectRatio > 1:
                k = imgSize / h
                wCal = math.ceil(k * w)
                imgResize = cv2.resize(imgCrop, (wCal, imgSize))
                wGap = math.ceil((imgSize - wCal) / 2)
                imgWhite[:, wGap:wCal + wGap] = imgResize
            else:
                k = imgSize / w
                hCal = math.ceil(k * h)
                imgResize = cv2.resize(imgCrop, (imgSize, hCal))
                hGap = math.ceil((imgSize - hCal) / 2)
                imgWhite[hGap:hCal + hGap, :] = imgResize

            prediction, index = classifier.getPrediction(imgWhite, draw=False)
            print(prediction, index)

            cv2.rectangle(imgOutput, (x-offset, y-offset-70), 
                         (x-offset+400, y-offset+60-50), (0, 255, 0), cv2.FILLED)
            cv2.putText(imgOutput, labels[index], (x, y-30), 
                       cv2.FONT_HERSHEY_COMPLEX, 2, (0, 0, 0), 2)
            cv2.rectangle(imgOutput, (x-offset, y-offset),
                         (x + w + offset, y + h + offset), (0, 255, 0), 4)

            # cv2.imshow('ImageCrop', imgCrop)
            cv2.imshow('ImageWhite', imgWhite)

        except Exception as e:
            print(f"Processing error: {e}")

    cv2.imshow('Image', imgOutput)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
