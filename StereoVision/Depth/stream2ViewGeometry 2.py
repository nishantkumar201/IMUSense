import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt
import math
import time
import pickle
import sys
import os

def main():
    file_count = 0
    frame_count = 0
    create_F = False
    disparity_container  = []
    wls_disp_container = []

    ### Read Calibration datas from file ###
    try:
        calibration_data = loadCalibrationData('../StereoCalibration/stereoMap.xml')
    except:
        print('Couldnt read calibration data')
    else:
        print("CameraMatrixL:\n", calibration_data['CameraMatrixL'])
        CameraMatrixL = calibration_data['CameraMatrixL']
        print("CameraMatrixR:\n", calibration_data['CameraMatrixR'])
        stereoMapL_x = calibration_data['stereoMapL_x']
        stereoMapL_y = calibration_data['stereoMapL_y']
        stereoMapR_x = calibration_data['stereoMapR_x']
        stereoMapR_y = calibration_data['stereoMapR_y']
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
        Q = calibration_data['Q']
        baseline = -1/Q[3,2]
        print("Q\n", Q)
        focalLength = CameraMatrixL[0][0]
        print("approximate baseline:", baseline,'[mm]')


    ### Try connecting with two webcams randomly. If the connected webcams are not in correct order (left & right) then swap them
    while True:
        cam1 = cv.VideoCapture(0) # cam1 should be the left camera
        cam2 = cv.VideoCapture(1) # cam2 should be the right camera
        ret1, img1 = cam1.read()
        ret2, img2 = cam2.read()
        if ((ret1 == True) and (ret2 == True)):
            break
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
        if frame_count == 99:
            # try:
            #     # cv_file = cv.FileStorage('depth'+ str(file_count)+'.xml', cv.FILE_STORAGE_WRITE)
            #     # cv_file.write('disparity',disparity_container)
            #     # cv_file.write('wls_disp', wls_disp_container)
            #     # cv_file.release()
            # except:
            #     print("couldnt save data")
            # else:
            #     print("saved in depth"+ str(file_count)+'.xml')
            #     disparity_container = []
            #     wls_disp_container = []
            frame_count = 0
            file_count += 1
            # save file 
            # disparity_container
            # wls_disp_container
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
            # cv.imshow('imgL',imgL)
            # cv.imshow('imgR',imgR)
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
            
            # imgL_line = imgL
            # imgR_line = imgR

            # lines1 = cv.computeCorrespondEpilines(pts2.reshape(-1,1,2), 2,F)
            # lines1 = lines1.reshape(-1,3)
            # imgL_line,_ = drawlines(imgL_line,imgR_line,lines1,pts1,pts2)

            # lines2 = cv.computeCorrespondEpilines(pts1.reshape(-1,1,2), 1,F)
            # lines2 = lines2.reshape(-1,3)
            # imgR_line,_ = drawlines(imgR_line,imgL_line,lines2,pts2,pts1)

            # LR_lines = np.hstack((imgL_line, imgR_line))
            # cv.imshow('LR_lines',LR_lines)
            #### undistort the images ###
            undistortedL = cv.remap(imgL, stereoMapL_x, stereoMapL_y, cv.INTER_LANCZOS4, cv.BORDER_CONSTANT, 0)
            undistortedR = cv.remap(imgR, stereoMapR_x, stereoMapR_y, cv.INTER_LANCZOS4, cv.BORDER_CONSTANT, 0)
            
            # imgL_rectified = cv.warpPerspective(imgL_line, H1, (w1, h1))
            # imgR_rectified = cv.warpPerspective(imgR_line, H2, (w2, h2))

            undistortedL_gray = cv.cvtColor(undistortedL, cv.COLOR_BGR2GRAY)
            undistortedR_gray = cv.cvtColor(undistortedR, cv.COLOR_BGR2GRAY)
            # imgL_rectified_gray = cv.cvtColor(imgL_rectified, cv.COLOR_BGR2GRAY)
            # imgR_rectified_gray = cv.cvtColor(imgR_rectified, cv.COLOR_BGR2GRAY)

            # create disparity map
            left_matcher = cv.StereoSGBM.create(numDisparities=96, blockSize=5, P1=400, P2=1600, speckleRange=2 )
            # right_matcher = cv.ximgproc.createRightMatcher(left_matcher) 
            # stereo = cv.StereoBM.create(numDisparities=160, blockSize=11)
            disparity_from_undistorted = left_matcher.compute(undistortedL_gray, undistortedR_gray)
            
            # right_disp = right_matcher.compute(undistortedR_gray, undistortedL_gray);

            ### for gray disparity map ###
            norm_image = cv.normalize(disparity_from_undistorted, None, alpha = 0, beta = 1.0, norm_type=cv.NORM_MINMAX, dtype=cv.CV_32F)

            ### for coloring the disparity map
            # norm_image = cv.normalize(disparity_from_undistorted, None, alpha = 0, beta = 255, norm_type=cv.NORM_MINMAX, dtype=cv.CV_8UC1)
            # norm_colormap = cv.applyColorMap(norm_image, cv.COLORMAP_DEEPGREEN)

            

            # wls_filter = cv.ximgproc.createDisparityWLSFilter(left_matcher);
            # filtered_disp = wls_filter.filter(disparity_from_undistorted, undistortedL_gray, disparity_map_right=right_disp);
            
            # disparity_container.append(disparity_from_undistorted)
            # wls_disp_container.append(filtered_disp)
            # norm_filtered = cv.normalize(filtered_disp, None, alpha=0, beta=1.0, norm_type=cv.NORM_MINMAX, dtype=cv.CV_32F)

            # cv.imshow('norm_filtered', norm_filtered)
            

            # display relavent images on screen
            # LeftRight = np.hstack((imgL,imgR))
            # cv.imshow('LeftRight', LeftRight)
            depth = baseline*focalLength/disparity_from_undistorted
            

            undistortedLR = np.hstack((undistortedL, undistortedR))
            cv.imshow('undistortedLR',undistortedLR)
            print(depth)
            cv.waitKey(0)

            # LR_rectified = np.hstack((imgL_rectified, imgR_rectified))
            # cv.imshow('LR_rectified', LR_rectified)

            # cv.imshow('disparity', disparity_from_rectified)
            cv.imshow('norm_image', norm_image)
            # cv.waitKey(0)
            # cv.imshow('norm_colormap', norm_colormap)
            # disparities = np.hstack((norm_filtered, norm_image))
            # cv.imshow('disparities', disparities)
 

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
        print("img1.shape:", img1.shape)
        print("img2.shape:", img2.shape)
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
