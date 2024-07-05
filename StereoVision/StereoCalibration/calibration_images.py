import cv2
import numpy as np




cam1 = cv2.VideoCapture(1)
cam2 = cv2.VideoCapture(2)

print("Please check the camera order")
while True:
    ret1 = cam1.grab()
    ret2 = cam2.grab()
    flag1, img1 = cam1.retrieve()
    flag2, img2 = cam2.retrieve()
    img1img2 = np.hstack((img1,img2))
    cv2.imshow('img1img2', img1img2)

    if cv2.waitKey(1) & 0xFF == ord('q'): 
        cv2.destroyWindow('img1img2')
        break

### Check whether to swap cameras or not ###

while True:
    should_swap = input("is the camera connected in right order? [y,n]: ")
    if should_swap == 'y':
        print("Proceeding with current camera setup")
        cam1.release()
        cam2.release()
        left = cv2.VideoCapture(1)
        right = cv2.VideoCapture(2)
        break
    elif should_swap == 'n':
        print("Proceeding with swapped camera setup")
        cam1.release()
        cam2.release()
        left = cv2.VideoCapture(2)
        right = cv2.VideoCapture(1)    
        break  

num = 0
##################
while left.isOpened():

    succes1, img = left.read()
    succes2, img2 = right.read()


    k = cv2.waitKey(5)

    if k == ord('q'):
        break
    elif k == ord('s'): # wait for 's' key to save and exit
        cv2.imwrite('stereoLeft/imageL' + str(num) + '.png', img)
        cv2.imwrite('stereoRight/imageR' + str(num) + '.png', img2)
        print("images saved!")
        num += 1

    leftright = np.hstack((img,img2))
    # cv2.imshow('left',img)
    # cv2.imshow('right',img2)

    cv2.imshow('LeftRight', leftright)