from ultralytics import YOLO
import matplotlib.pyplot as plt
# from PIL import Image
import numpy as np
import cv2
import os
from sklearn.decomposition import PCA
import datetime

def save_data_with_timestamp(data):
    # Generate timestamp
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    # Open the text file in append mode
    with open('data_with_timestamp.txt', 'a') as file:
        # Write timestamp and data to the file
        file.write(f"{timestamp}: {data}\n")

############Is contour inside another Contours##########
def is_contour_inside(contour1, contour2):
    for point in contour2:
        if cv2.pointPolygonTest(contour1, point[0], False) < 0:
            return False
    return True
############Is contour inside another Contours##########

print("load model")
model = YOLO("segment.pt")
cap = cv2.VideoCapture("mersivityball1_short.mp4")
# cap = cv2.VideoCapture(1)
if not cap.isOpened():
    print("Error: Unable to open camera.")
    exit()

while True:
    ret, frame = cap.read()
    original_frame = frame
    black_background = np.zeros_like(frame)
    ############segmentation##########
    segmentation = model.predict(frame, conf = 0.52)
    segmentation_img = segmentation[0].plot()
    ############segmentation##########

    ############Green Pixels##########
    hsv_image = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    lower_green = np.array([60, 60, 60]) # mersivityball(60,60,60)
    upper_green = np.array([110, 110, 120]) # mersivityball(110,110,120)


    mask = cv2.inRange(hsv_image, lower_green, upper_green)
    kernel = np.ones((5, 5), np.uint8)
    # mask = cv2.erode(mask, kernel, iterations=1)
    # cv2.imshow('mask', mask)
    mask = cv2.dilate(mask, kernel, iterations=10)
    # mask = cv2.erode(mask, kernel, iteration`s=1)
    # mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel=(9,9), iterations=16) # Richard: adding morphorlogical closing 
    # cv2.imshow('morphological_closing', mask)
    # cv2.imshow('mask', mask)
    green_highlighted_image = cv2.bitwise_and(frame, frame, mask=mask)
    ############Green Pixels##########

    


    ############Print the green XY coor##########
    '''
    green_indices = np.argwhere(mask > 0)
    green_coordinates = green_indices[:, [1, 0]]  # Swap columns to get (x, y) format
    for x, y in green_coordinates:
        print("Pixel (x, y):", x, y)
    '''
    ############Print the green XY coor##########

    ############Green Countors##########
    green_contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    temp = green_contours
    green_contours = [contour.astype(np.float32) for contour in green_contours]
  


    ############Green Countors##########
    list = []
    ############Checks if the green Countors are in the human segmentation##########
    if(len(segmentation[0])):
        masks = segmentation[0].masks.xy    
        for i in range(len(segmentation[0].boxes)):
            if(segmentation[0].boxes.cls[i].item() == 0):
                for j in range(len(green_contours)):   
                    if is_contour_inside(segmentation[0].masks.xy[i], green_contours[j]):
                        print("Tape {} detected".format(j))
                        list.append(j)
                        
                        

    ############Checks if the green Countors are in the human segmentation##########
       
    for i in range(len(temp)): 
        if(i in list):
            cnt = temp [i]           
            cv2.drawContours(black_background, [cnt], -1, (0, 255, 0), 2)
            M = cv2.moments(cnt)
            if M["m00"] != 0:
                centroid_x = int(M["m10"] / M["m00"])
                centroid_y = int(M["m01"] / M["m00"])
                centroid = (centroid_x, centroid_y)
                # Check if there are any contours found
            
            # Convert contours to PCA-ready format
            contour_data = np.vstack([cnt.reshape(-1, 2) ])

            # Apply PCA
            pca = PCA(n_components=2)  # Choose number of components as required
            pca.fit(contour_data)

            # Get principal components
            components = pca.components_

            # Calculate endpoints for principal components
            length = 100  # Length of the lines representing the principal components
            endpoint1 = (components[0] * length).astype(int)  # First principal component
            endpoint2 = (components[1] * length).astype(int)  # Second principal component

            # Draw principal components on the frame
            center = (int(frame.shape[1] / 2), int(frame.shape[0] / 2))  # Center of the frame

 
            endpoint1_coords = (centroid[0] + endpoint1[0], centroid[1] + endpoint1[1])
            endpoint2_coords = (centroid[0] + endpoint2[0], centroid[1] + endpoint2[1])
            cv2.line(black_background, centroid, endpoint1_coords, (0, 0, 255), 2)  # Red line for first principal component
            # cv2.line(black_background, centroid, endpoint2_coords, (0, 255, 0), 2)  # Green line for second principal component
            data = "contour" + str(j) + ":"  + str(centroid) + ":" + str(endpoint1_coords) +":"+ str(endpoint2_coords)
            save_data_with_timestamp(data)    
            
            stack1 = np.hstack((original_frame,segmentation_img))
            stack2 = np.hstack((green_highlighted_image, black_background))
            stacks = np.hstack((stack1, stack2))
            cv2.imshow('Final results', stacks)
        else:
            print("No green contours found.")

    # cv2.imshow('All Green', green_highlighted_image)
    # cv2.imshow('Segment', segmentation_img)
    # cv2.imshow('outlines', black_background)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    if cv2.waitKey(1) & 0xFF == ord('m'):
        data = "-------------------------------------------------------------------"
        save_data_with_timestamp(data)   

cap.release()
cv2.destroyAllWindows()