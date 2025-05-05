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