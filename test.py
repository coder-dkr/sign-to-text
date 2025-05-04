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

labels = ["Hello", "I love you", "No", "Okay", "Please", "Thank you", "Yes","F#ck Off"]

while True:
    success, img = cap.read()
    if not success:
        continue
        
    imgOutput = img.copy()
    hands, img = detector.findHands(img)
    
    if hands:
        hand = hands[0]
        x, y, w, h = hand['bbox']
        
        # Ensure the crop coordinates are within image bounds
        imgHeight, imgWidth = img.shape[:2]
        x1 = max(0, x - offset)
        y1 = max(0, y - offset)
        x2 = min(imgWidth, x + w + offset)
        y2 = min(imgHeight, y + h + offset)
        
        imgCrop = img[y1:y2, x1:x2]
        
        # Skip if the crop area is too small
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

            # Display results
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




# GREAT BARRIER

# import cv2
# from cvzone.HandTrackingModule import HandDetector
# from cvzone.ClassificationModule import Classifier
# import numpy as np
# import math
# import tensorflow as tf  # Explicit import

# offset = 20
# imgSize = 300  # This was missing in your code
# counter = 0

# # Initialize
# cap = cv2.VideoCapture(0)
# detector = HandDetector(maxHands=1)

# # Verify TensorFlow is working
# print("TensorFlow version:", tf.__version__)
# print("Keras version:", tf.keras.__version__)

# # Load model (with error handling)
# try:
#     classifier = Classifier("Model/keras_model.h5", "Model/labels.txt")
# except Exception as e:
#     print("Error loading model:", e)
#     exit()

# # Your labels
# labels = ["Hello", "I love you", "No", "Okay", "Please", "Thank you", "Yes"]

# # Main loop
# while True:
#     success, img = cap.read()
#     if not success:
#         continue
        
#     imgOutput = img.copy()
#     hands, img = detector.findHands(img)
    
#     if hands:
#         hand = hands[0]
#         x, y, w, h = hand['bbox']
        
#         # Create white background
#         imgWhite = np.ones((imgSize, imgSize, 3), np.uint8) * 255
#         imgCrop = img[y-offset:y+h+offset, x-offset:x+w+offset]
        
#         # Resize logic (same as yours)
#         aspectRatio = h/w
#         if aspectRatio > 1:
#             # Vertical image
#             k = imgSize / h
#             wCal = math.ceil(k * w)
#             imgResize = cv2.resize(imgCrop, (wCal, imgSize))
#             wGap = math.ceil((imgSize - wCal)/2)
#             imgWhite[:, wGap:wCal+wGap] = imgResize
#         else:
#             # Horizontal image
#             k = imgSize / w
#             hCal = math.ceil(k * h)
#             imgResize = cv2.resize(imgCrop, (imgSize, hCal))
#             hGap = math.ceil((imgSize - hCal)/2)
#             imgWhite[hGap:hCal+hGap, :] = imgResize

#         # Get prediction
#         try:
#             prediction, index = classifier.getPrediction(imgWhite)
#             cv2.putText(imgOutput, labels[index], (x, y-30), 
#                        cv2.FONT_HERSHEY_COMPLEX, 2, (0,0,0), 2)
#         except Exception as e:
#             print("Prediction error:", e)

#     cv2.imshow('Output', imgOutput)
#     if cv2.waitKey(1) & 0xFF == ord('q'):
#         break

# cap.release()
# cv2.destroyAllWindows()