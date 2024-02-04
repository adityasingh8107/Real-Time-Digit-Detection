#!/usr/bin/env python
# coding: utf-8

# In[21]:


import cv2
import numpy as np
import torch
from torchvision import transforms
from CNN_Model import CNN


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = CNN()
model.load_state_dict(torch.load('weights1.pth', map_location=device))
model.eval()
model = model.to(device)


hsv_value = np.load('hsv_value.npy')

cap = cv2.VideoCapture(0)
cap.set(3, 1280)
cap.set(4, 720)

kernel = np.ones((5, 5), np.uint8)

canvas = np.zeros((720, 1280, 3))

x1 = 0
y1 = 0

noise_thresh = 800
i=1

while i:
    _, frame = cap.read()

    if canvas is not None:
        canvas = np.zeros_like(frame)

    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    
    lower_range = hsv_value[0]
    upper_range = hsv_value[1]

    
    mask = cv2.inRange(hsv, lower_range, upper_range)
    mask = cv2.erode(mask, kernel, iterations=1)
    mask = cv2.dilate(mask, kernel, iterations=2)

    contours, hierarchy = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if contours and cv2.contourArea(max(contours, key=cv2.contourArea)) > noise_thresh:
        c = max(contours, key=cv2.contourArea)
        x2, y2, w, h = cv2.boundingRect(c)

        if x1 == 0 and y1 == 0:
            x1, y1 = x2, y2
        else:
            digit_roi = frame[y2:y2 + h, x2:x2 + w]
            digit_roi_gray = cv2.cvtColor(digit_roi, cv2.COLOR_BGR2GRAY)
            digit_roi_gray = cv2.resize(digit_roi_gray, (28, 28))

            transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
            digit_tensor = transform(digit_roi_gray).unsqueeze(0).to(device)

            with torch.no_grad():
                output = model(digit_tensor)

            prediction = torch.argmax(output, 1).item()
            cv2.rectangle(frame, (x2, y2), (x2 + w, y2 + h), (255, 0, 0), 2)
            cv2.putText(frame, f"Predicted Digit: {prediction}", (x2, y2 - 10), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)

            canvas = cv2.line(canvas, (x1, y1), (x2, y2), [0, 255, 0], 4)

        x1, y1 = x2, y2

    else:
        x1, y1 = 0, 0

    frame = cv2.add(canvas, frame)

    stacked = np.hstack((canvas, frame))
    cv2.imshow('Screen_Pen', cv2.resize(stacked, None, fx=0.6, fy=0.6))

    if cv2.waitKey(100) == 10:
        break

    if cv2.waitKey(1) & 0xFF == ord('c'):
        canvas = None
        i=0
        
cv2.destroyAllWindows()
cap.release()    




# In[ ]:




