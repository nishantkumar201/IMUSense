import serial.tools.list_ports

import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
import cv2
import os
import time

def find_red_pixels(image):
    # Define lower and upper bounds for red color in HSV
    lower_green = np.array([35, 100, 100])
    upper_green = np.array([85, 255, 255])

    # Convert image to HSV color space
    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # Threshold the HSV image to get only red colors
    mask = cv2.inRange(hsv_image, lower_green, upper_green)

    
    # Find contours in the mask
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if len(contours) > 0:
        return frame,True
    return frame,False

cap = cv2.VideoCapture(0)
cap2 = cv2.VideoCapture(1)
if not cap.isOpened():
    print("Error: Unable to open camera.")
    exit()

if not cap2.isOpened():
    print("Error: Unable to open camera.")
    exit()


ports = serial.tools.list_ports
serialInst = serial.Serial()

portVar = "COM" + str(3)
serialInst.baudrate =  9600
serialInst.port = portVar
serialInst.open()

numleds = 10
count = 0
pattern = "0"
output_dir = 'captured_frames'
os.makedirs(output_dir, exist_ok=True)

Finished = "finish"

frame_count = 0
ret, frame = cap.read()
skiponce = 0
while True:
    ret, frame = cap2.read()
    message = input("Enter message to send to Arduino: ")
    skiponce = 0

        
        
    if(message == "exit"):
        exit()
    elif(message == "next" and skiponce ==0):
        for i in range(numleds):
            serialInst.write(message.encode('utf-8'))
            count = count +1
            time.sleep(2)
            ret, frame = cap2.read()
            processedframe, hi = find_red_pixels(frame)
            cv2.imshow('Red Pixels Detection', processedframe)
            # Break the loop if 'q' is pressed
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

            if(hi):
                pattern = pattern + str(1)
            else:
                pattern = pattern + str(0)

        serialInst.write(pattern.encode('utf-8'))
        count = 0
        pattern = ""
        time.sleep(2)
        ret, frame = cap.read()
        
        frame_filename = os.path.join(output_dir, f'frame_{frame_count}.jpg')
        frame_count = frame_count +1
        cv2.imwrite(frame_filename, frame)
        skiponce = 1

            
    elif(message == "pattern"):
        print(pattern)
    elif(message == "swap"):
        cap, cap2 = cap2, cap
    elif(message == "camera"):
        ret, frame = cap.read()
        
        frame_filename = os.path.join(output_dir, f'frame_{frame_count}.jpg')
        frame_count = frame_count +1
        cv2.imwrite(frame_filename, frame)
        

        
