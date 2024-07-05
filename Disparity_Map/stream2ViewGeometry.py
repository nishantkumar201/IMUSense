import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt
import math
import time
import pickle
import sys


def main():
    
    create_F = True

    ### Read Calibration datas from file ###
    # try:
        # calibration_data = loadCalibrationData('StereoCalibration/stereoMap.xml')
    # except:
        # print('Couldnt read calibration data')
    # else:
        # print("CameraMatrixL:\n", calibration_data['CameraMatrixL'])
        # print("CameraMatrixR:\n", calibration_data['CameraMatrixR'])
        # stereoMapL_x = calibration_data['stereoMapL_x']
        # stereoMapL_y = calibration_data['stereoMapL_y']
        # stereoMapR_x = calibration_data['stereoMapR_x']
        # stereoMapR_y = calibration_data['stereoMapR_y']
        # print("stereroMapL_x:\n", stereoMapL_x)
        # print("stereroMapL_y:\n", stereoMapL_y)
        # print("stereroMapR_x:\n", stereoMapR_x)
        # print("stereroMapR_y:\n", stereoMapR_y)
        # print("CameraMatrixL:\n", calibration_data['CameraMatrixL'])
        # print("CameraMatrixR:\n", calibration_data['CameraMatrixR'])
        # stereoMapL1 = calibration_data['stereoMapL1']
        # stereoMapL2 = calibration_data['stereoMapL2']
        # stereoMapR1 = calibration_data['stereoMapR1']
        # stereoMapR2 = calibration_data['stereoMapR2']
        # Q = calibration_data['Q']


    ### Try connecting with two webcams randomly. If the connected webcams are not in correct order (left & right) then swap them
    cam1 = cv.VideoCapture(0) # cam1 should be the left camera
    cam2 = cv.VideoCapture(1) # cam2 should be the right camera
    left, right = checkCameraOrder(cam1, cam2)

    ### make sure new camera connections are secure ###
    retries = 5
    while not(left.isOpened()):
        retries -= 1
        time.sleep(0.1)
        if retries == 0:
            print("failed to connect to left")
            return 0
    print("left connected")
    retries = 5
    while not(right.isOpened()):
        retries -= 1
        time.sleep(0.1)
        if retries == 0:
            print("failed to connect to right")
            return 0
    print("right connected")

    first = True
    ### Now we should have correct camera orientation ###
    connectionChecked = False
    print("Confirm the Camera order with 'q'")
    while(True):  
            # Capture the video frame by frame 
        ret1 = left.grab()
        ret2 = right.grab() 
        while not(ret1 and ret2):
            ret1 = left.grab()
            ret2 = right.grab() 
        flag1, imgL = left.retrieve()
        flag2, imgR = right.retrieve()
        while not(flag1 and flag2):
            flag1, imgL = left.retrieve()
            flag2, imgR = right.retrieve()
        if (connectionChecked == False):  
            img1img2 = np.hstack((imgL,imgR))
            cv.imshow('img1img2', img1img2)
            if cv.waitKey(1) & 0xFF == ord('q'):
                connectionChecked = True
                cv.destroyWindow('img1img2')
                print("connection Checked")
                continue
        else : # connectionChecked == True
            # imgLimgR = np.hstack((imgL,imgR))
            # cv.imshow('imgLimgR', imgLimgR)
            if create_F:
                create_F = False
                h1, w1, _ = imgL.shape
                h2, w2, _ = imgR.shape
                F, mask, pts1, pts2 = estimateFundamentalUncalibrated(imgL, imgR)
                
                # We select only inlier points
                pts1 = pts1[mask.ravel()==1]
                pts2 = pts2[mask.ravel()==1]

                _, H1, H2 = cv.stereoRectifyUncalibrated(
                    np.float32(pts1), np.float32(pts2), F, imgSize=(w1, h1)
                )
            
            imgL_line = imgL
            imgR_line = imgR

            # lines1 = cv.computeCorrespondEpilines(pts2.reshape(-1,1,2), 2,F)
            # lines1 = lines1.reshape(-1,3)
            # imgL_line,_ = drawlines(imgL_line,imgR_line,lines1,pts1,pts2)

            # lines2 = cv.computeCorrespondEpilines(pts1.reshape(-1,1,2), 1,F)
            # lines2 = lines2.reshape(-1,3)
            # imgR_line,_ = drawlines(imgR_line,imgL_line,lines2,pts2,pts1)

            # LR_lines = np.hstack((imgL_line, imgR_line))
            # cv.imshow('LR_lines',LR_lines)
            #### undistort the images ###
            # undistortedL = cv.remap(imgL, stereoMapL_x, stereoMapL_y, cv.INTER_LANCZOS4, cv.BORDER_CONSTANT, 0)
            # undistortedR = cv.remap(imgR, stereoMapR_x, stereoMapR_y, cv.INTER_LANCZOS4, cv.BORDER_CONSTANT, 0)
            
            imgL_rectified = cv.warpPerspective(imgL_line, H1, (w1, h1))
            imgR_rectified = cv.warpPerspective(imgR_line, H2, (w2, h2))

            # undistortedL_gray = cv.cvtColor(undistortedL, cv.COLOR_BGR2GRAY)
            # undistortedR_gray = cv.cvtColor(undistortedR, cv.COLOR_BGR2GRAY)
            imgL_rectified_gray = cv.cvtColor(imgL_rectified, cv.COLOR_BGR2GRAY)
            imgR_rectified_gray = cv.cvtColor(imgR_rectified, cv.COLOR_BGR2GRAY)

            # create disparity map
            stereo = cv.StereoSGBM.create(numDisparities=64, blockSize=15)
            # stereo = cv.StereoBM.create(numDisparities=32, blockSize=15)
            # disparity_from_undistorted = stereo.compute(undistortedL_gray, undistortedR_gray)
            # norm_image = cv.normalize(disparity_from_undistorted, None, alpha = 0, beta = 255, norm_type=cv.NORM_MINMAX, dtype=cv.CV_32F)
            disparity_from_rectified = stereo.compute(imgL_rectified_gray, imgR_rectified_gray)
            norm_image = cv.normalize(disparity_from_rectified, None, alpha = 0, beta = 1, norm_type=cv.NORM_MINMAX, dtype=cv.CV_32F)
            

            # print("disparity_from_rectified:\n", disparity_from_rectified)
            # print("norm_from_rectified:\n", norm_image)

            ### now use reprojectImageTo3D() to get the depth ### 
            # depth_map = cv.reprojectImageTo3D()


            # display relavent images on screen
            # LeftRight = np.hstack((imgL,imgR))
            # cv.imshow('LeftRight', LeftRight)

            # undistortedLR = np.hstack((undistortedL, undistortedR))
            # cv.imshow('undistortedLR',undistortedLR)

            # LR_rectified = np.hstack((imgL_rectified, imgR_rectified))
            # cv.imshow('LR_rectified', LR_rectified)

            # cv.imshow('disparity', disparity_from_rectified)
            cv.imshow('norm_image', norm_image)
            
            # cv.imshow('undistortedL',undistortedL)
            # cv.imshow('undistortedR',undistortedR)
            # print(norm_image.dtype)
            # print(norm_image.shape)
            # print(norm_image)
            # cv.waitKey(0)

            if cv.waitKey(1) & 0xFF == ord('q'): 
                # cv.destroyWindow('norm_image')
                # cv.destroyWindow('img1img2_rectified')
                cv.destroyAllWindows()
                break
    
    # After the loop release the cap object 
    left.release() 
    right.release()
    # Destroy all the windows 
    cv.destroyAllWindows() 

def calculateFundamentalMatrix(img1, img2, sift):
    # find the keypoints and descriptors with SIFT
    kp1, des1 = sift.detectAndCompute(img1,None)
    kp2, des2 = sift.detectAndCompute(img2,None)
    print(len(kp1))
    print(len(kp2))
    # FLANN parameters
    FLANN_INDEX_KDTREE = 1
    index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
    search_params = dict(checks=50)
    flann = cv.FlannBasedMatcher(index_params,search_params)
    matches = flann.knnMatch(des1,des2,k=2)

    pts1 = []
    pts2 = []
    
    # ratio test as per Lowe's paper
    for i,(m,n) in enumerate(matches):
        if m.distance < 0.8*n.distance:
            pts2.append(kp2[m.trainIdx].pt)
            pts1.append(kp1[m.queryIdx].pt)


    ### Let's find the Fundamental Matrix. ###
    pts1 = np.int32(pts1)
    pts2 = np.int32(pts2)

    print("len(pts1):", len(pts1))
    F, mask = cv.findFundamentalMat(pts1,pts2,cv.FM_RANSAC, 10) # cv.FM_LMEDS
    return F, mask, pts1, pts2

def drawlines(img1,img2,lines,pts1,pts2):
    ''' img1 - image on which we draw the epilines for the points in img2
    lines - corresponding epilines '''
    r,c,_ = img1.shape

    for r,pt1,pt2 in zip(lines,pts1,pts2):
        color = tuple(np.random.randint(0,255,3).tolist())
        x0,y0 = map(int, [0, -r[2]/r[1] ])
        x1,y1 = map(int, [c, -(r[2]+r[0]*c)/r[1] ])
        img1 = cv.line(img1, (x0,y0), (x1,y1), color,1)
        img1 = cv.circle(img1,tuple(pt1),5,color,-1)
        img2 = cv.circle(img2,tuple(pt2),5,color,-1)
    return img1,img2

def loadCalibrationData(filepath):
    ''' Load stored stereoCalibration result matrix'''
    ret_dict = {} # initialize dictionary 
    cv_file = cv.FileStorage()
    cv_file.open(filepath, cv.FileStorage_READ)
    ret_dict.update({'stereoMapL_x': cv_file.getNode('stereoMapL_x').mat()})
    ret_dict.update({'stereoMapL_y': cv_file.getNode('stereoMapL_y').mat()})
    ret_dict.update({'stereoMapR_x': cv_file.getNode('stereoMapR_x').mat()})
    ret_dict.update({'stereoMapR_y': cv_file.getNode('stereoMapR_y').mat()})
    ret_dict.update({'CameraMatrixL': cv_file.getNode('newCameraMatrixL').mat()})
    ret_dict.update({'CameraMatrixR': cv_file.getNode('newCameraMatrixR').mat()})
    ret_dict.update({'distL': cv_file.getNode('distL').mat()})
    ret_dict.update({'distR': cv_file.getNode('distR').mat()})
    ret_dict.update({'rot': cv_file.getNode('rot').mat()})
    ret_dict.update({'trans': cv_file.getNode('trans').mat()})
    ret_dict.update({'essentialMatrix': cv_file.getNode('essentialMatrix').mat()})
    ret_dict.update({'fundamentalMatrix': cv_file.getNode('fundamentalMatrix').mat()})
    ret_dict.update({'Q': cv_file.getNode('Q').mat()})

    return ret_dict

def checkCameraOrder(cam1, cam2):
    ''' Confirm cameras are connected in desired order. Swap them if not'''
    print("Please check the camera order") ### Check whether to swap cameras or not 
    while True:
        ret1 = cam1.grab()
        ret2 = cam2.grab()
        flag1, img1 = cam1.retrieve()
        flag2, img2 = cam2.retrieve()
        img1img2 = np.hstack((img1,img2))
        cv.imshow('img1img2', img1img2)

        if cv.waitKey(1) & 0xFF == ord('q'): 
            cv.destroyWindow('img1img2')
            break

    while True:
        should_swap = input("is the camera connected in right order? [y,n]: ")
        if should_swap == 'y':
            print("Proceeding with current camera setup")
            cam1.release()
            cam2.release()
            left = cv.VideoCapture(0)
            right = cv.VideoCapture(1)
            break
        elif should_swap == 'n':
            print("Proceeding with swapped camera setup")
            cam1.release()
            cam2.release()
            left = cv.VideoCapture(1)
            right = cv.VideoCapture(0)    
            break  

    return left, right 

def estimateFundamentalUncalibrated(imgL, imgR):
    ''' estimate the fundamental matrix '''
    sift = cv.SIFT_create(5000)
    print("create_F")
    h1, w1, _ = imgL.shape
    h2, w2, _ = imgR.shape
    
    # find the keypoints and descriptors with SIFT
    kp1, des1 = sift.detectAndCompute(imgL,None)
    kp2, des2 = sift.detectAndCompute(imgR,None)
    
    # FLANN parameters
    FLANN_INDEX_KDTREE = 1
    index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
    search_params = dict(checks=50)
    flann = cv.FlannBasedMatcher(index_params,search_params)
    matches = flann.knnMatch(des1,des2,k=2)
    pts1 = []
    pts2 = []
    for i,(m,n) in enumerate(matches):
        if m.distance < 0.8*n.distance:
            pts2.append(kp2[m.trainIdx].pt)
            pts1.append(kp1[m.queryIdx].pt)
    pts1 = np.int32(pts1)
    pts2 = np.int32(pts2)
    F, mask = cv.findFundamentalMat(pts1,pts2,cv.FM_RANSAC, 3, 0.98) # cv.FM_LMEDS

    return F, mask, pts1, pts2
                


if __name__ == "__main__":
    print("Calibrate stereo system before calculating disparity")
    main()
