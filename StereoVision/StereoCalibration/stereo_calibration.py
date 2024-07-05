import numpy as np
import cv2 as cv
import glob
import pickle as pkl



################ FIND CHESSBOARD CORNERS - OBJECT POINTS AND IMAGE POINTS #############################

chessboardSize = (6,4)
frameSize = (1280,720)



# termination criteria
criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)


# prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
objp = np.zeros((chessboardSize[0] * chessboardSize[1], 3), np.float32)
objp[:,:2] = np.mgrid[0:chessboardSize[0],0:chessboardSize[1]].T.reshape(-1,2)
objp = objp * 30 # 30mm per points
print(objp)


# Arrays to store object points and image points from all the images.
objpoints = [] # 3d point in real world space
imgpointsL = [] # 2d points in image plane.
imgpointsR = [] # 2d points in image plane.


imagesLeft = glob.glob('stereoLeft/*.png')
imagesLeft.sort()
imagesRight = glob.glob('stereoRight/*.png')
imagesRight.sort()
print(imagesLeft)
print(imagesRight)

for imgLeft, imgRight in zip(imagesLeft, imagesRight):

    imgL = cv.imread(imgLeft)
    imgR = cv.imread(imgRight)
    grayL = cv.cvtColor(imgL, cv.COLOR_BGR2GRAY)
    grayR = cv.cvtColor(imgR, cv.COLOR_BGR2GRAY)

    # Find the chess board corners
    retL, cornersL = cv.findChessboardCorners(grayL, chessboardSize, None)
    retR, cornersR = cv.findChessboardCorners(grayR, chessboardSize, None)

    # If found, add object points, image points (after refining them)
    if retL and retR == True:

        objpoints.append(objp)

        cornersL = cv.cornerSubPix(grayL, cornersL, (11,11), (-1,-1), criteria) # refined corners
        imgpointsL.append(cornersL)

        cornersR = cv.cornerSubPix(grayR, cornersR, (11,11), (-1,-1), criteria) # refined corners
        imgpointsR.append(cornersR)

        # Draw and display the corners
        cv.drawChessboardCorners(imgL, chessboardSize, cornersL, retL)
        cv.drawChessboardCorners(imgR, chessboardSize, cornersR, retR)
        LeftRight = np.hstack((imgL,imgR))
        cv.imshow('LeftRight', LeftRight)
        print("imgLeft:", imgLeft)
        print("imgRight:", imgRight)
        print()
        cv.waitKey(500)


cv.destroyAllWindows()




############## CALIBRATION #######################################################
# take the calibration data from ./CameraCalibration/Left and ./CameraCalibration/Right
# retL, cameraMatrixL, distL, rvecsL, tvecsL = cv.calibrateCamera(objpoints, imgpointsL, frameSize, None, None)
# print('RMS reprojection error of left camera calibration:', retL)
with open('../CameraCalibration/Left/Left_calibration.pkl', 'rb') as pickle_file_L:
    cameraMatrixL, distL = pkl.load(pickle_file_L)
print()    
print("cameraMatrixL, distL\n",cameraMatrixL, distL)
heightL, widthL, channelsL = imgL.shape
newCameraMatrixL, roi_L = cv.getOptimalNewCameraMatrix(cameraMatrixL, distL, (widthL, heightL), 1, (widthL, heightL))

# retR, cameraMatrixR, distR, rvecsR, tvecsR = cv.calibrateCamera(objpoints, imgpointsR, frameSize, None, None)
with open('../CameraCalibration/Right/Right_calibration.pkl', 'rb') as pickle_file_R:
    cameraMatrixR, distR = pkl.load(pickle_file_R)
print()
print("cameraMatrixR, distR\n",cameraMatrixR, distR)
heightR, widthR, channelsR = imgR.shape
# print('RMS reprojection error of right camera calibration:', retR)
newCameraMatrixR, roi_R = cv.getOptimalNewCameraMatrix(cameraMatrixR, distR, (widthR, heightR), 1, (widthR, heightR))


########## Stereo Vision Calibration #############################################

flags = 0
flags |= cv.CALIB_FIX_INTRINSIC
# Here we fix the intrinsic camara matrixes so that only Rot, Trns, Emat and Fmat are calculated.
# Hence intrinsic parameters are the same 

criteria_stereo= (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)
print("grayL.shape[::-1]:", grayL.shape[::-1])
# This step is performed to transformation between the two cameras and calculate Essential and Fundamenatl matrix
retStereo, newCameraMatrixL, distL, newCameraMatrixR, distR, rot, trans, essentialMatrix, fundamentalMatrix = cv.stereoCalibrate(objpoints, imgpointsL, imgpointsR, newCameraMatrixL, distL, newCameraMatrixR, distR, grayL.shape[::-1], criteria_stereo, flags)
# retStereo, newCameraMatrixL, distL, newCameraMatrixR, distR, rot, trans, essentialMatrix, fundamentalMatrix = cv.stereoCalibrate(objpoints, imgpointsL, imgpointsR, cameraMatrixL, distL, cameraMatrixR, distR, grayL.shape[::-1], criteria_stereo, flags)

print("newCameraMatrixL:\n", newCameraMatrixL)
print("newCameraMatrixR:\n", newCameraMatrixR)

# ########## Stereo Rectification #################################################

rectifyScale= 1 # 1
rectL, rectR, projMatrixL, projMatrixR, Q, roi_L, roi_R= cv.stereoRectify(newCameraMatrixL, distL, newCameraMatrixR, distR, grayL.shape[::-1], rot, trans, alpha=rectifyScale, newImageSize=(0,0))
stereoMapL = cv.initUndistortRectifyMap(newCameraMatrixL, distL, rectL, projMatrixL, grayL.shape[::-1], cv.CV_16SC2)
stereoMapR = cv.initUndistortRectifyMap(newCameraMatrixR, distR, rectR, projMatrixR, grayR.shape[::-1], cv.CV_16SC2)

print("roi_L:", roi_L)
print("roi_R:",roi_R)
# testing mapping
imgL = cv.imread('stereoLeft/imageL0.png')
imgR = cv.imread('stereoRight/imageR0.png')
# print("roi_L:\n", roi_L)
# print("roi_R:\n", roi_R)
remappedL = cv.remap(imgL, stereoMapL[0], stereoMapL[1], cv.INTER_LANCZOS4, cv.BORDER_CONSTANT, 0)
remappedR = cv.remap(imgR, stereoMapR[0], stereoMapR[1], cv.INTER_LANCZOS4, cv.BORDER_CONSTANT, 0)
# remappedL_rectangle = cv.rectangle(remappedL, (roi_L[0],roi_L[2]), (roi_L[1],roi_L[3]), (0,255,0))
# remappedR_rectangle = cv.rectangle(remappedR, (roi_R[0],roi_R[2]), (roi_R[1],roi_R[3]), (0,255,0))
# cv.imshow('remappedL_rectangle', remappedL_rectangle)
# cv.imshow('remappedR_rectangle', remappedR_rectangle)
remappedLR = np.hstack((remappedL, remappedR))
cv.imshow('remappedLR', remappedLR)
cv.waitKey(0)
print("Saving parameters!")
try:
    cv_file = cv.FileStorage('stereoMap.xml', cv.FILE_STORAGE_WRITE)
    
    # cv_file.write('stereoMapL1',stereoMapL1)
    # cv_file.write('stereoMapL2',stereoMapL2)
    # cv_file.write('stereoMapR1',stereoMapR1)
    # cv_file.write('stereoMapR2',stereoMapR2)
    # cv_file.write('newCameraMatrixL',newCameraMatrixL)
    # cv_file.write('newCameraMatrixR',newCameraMatrixR)
    # cv_file.write('distL',distL)
    # cv_file.write('distR',distR)
    # cv_file.write('rot',rot)
    # cv_file.write('trans',trans)
    # cv_file.write('essentialMatrix',essentialMatrix)
    # cv_file.write('fundamentalMatrix',fundamentalMatrix)
    # cv_file.write('Q', Q)
    cv_file.write('stereoMapL_x',stereoMapL[0])
    cv_file.write('stereoMapL_y',stereoMapL[1])
    cv_file.write('stereoMapR_x',stereoMapR[0])
    cv_file.write('stereoMapR_y',stereoMapR[1])
    cv_file.write('newCameraMatrixL',newCameraMatrixL)
    cv_file.write('newCameraMatrixR',newCameraMatrixR)
    cv_file.write('distL',distL)
    cv_file.write('distR',distR)
    cv_file.write('rot',rot)
    cv_file.write('trans',trans)
    cv_file.write('essentialMatrix',essentialMatrix)
    cv_file.write('fundamentalMatrix',fundamentalMatrix)
    cv_file.write('Q', Q)

    cv_file.release()
except:
    print("Couldnt save calibration data")
else:
    print("calibration data saved")


