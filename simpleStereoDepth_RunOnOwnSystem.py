import numpy as np
import cv2
from matplotlib import pyplot as plt

imgL = cv2.imread('left.png',0)
imgR = cv2.imread('right.png',0)

## The below is the basic function, corrected as per the link above
## this function is implemented incorrectly on the openCV-Python online
## tutorial.
def stereo_disparity(Number_of_Disparities, Window_Size):
	stereo = cv2.StereoBM(cv2.STEREO_BM_BASIC_PRESET,ndisparities=Number_of_Disparities, SADWindowSize=Window_Size)
	##http://docs.opencv.org/2.4.1/modules/calib3d/doc/camera_calibration_and_3d_reconstruction.html#stereobm-stereobm
	disparity = stereo.compute(imgL,imgR)
	return disparity


### I'm doing a basic sweep of the Window Sizes.
### Note: the Disparities size can only be multiples of 16, hence param=16
param=16
x=np.arange(0,6)

for i in x:
	plt.subplot(2,np.ceil(len(x)/2).astype(np.uint32()),(i+1))
	plt.imshow(stereo_disparity(param*x[i], 7),'gray')
	plt.title("Disparities %s " %(param*x[i]))
plt.show()

### Doing a basic sweep for different Window Sizes below.
### Note: The Window Size variance isn't fully clear to me.
### The Window Size has the following restrictions: 
### (SADWindowSize must be odd, be within 5..255 and be not larger 
### than image width or height in function findStereoCorrespondenceBM)

param2=5
x=[25,27,29,31,33,35]

for i in range(0,len(x)):
        plt.subplot(2,np.ceil(len(x)/2).astype(np.uint32()),(i+1))
        plt.imshow(stereo_disparity(48,np.ceil(x[i]).astype(np.uint32())),'gray')
        plt.title("Window Size %s " %(np.ceil(x[i]).astype(np.uint32())))
plt.show()


##Below is the disparity map I thought was the best of the lot.

Disp=48
Win=31

plt.imshow(stereo_disparity(Disp,Win),'gray')
plt.title("Window Size=%s, Number of Disparities=%s" %(Win,Disp))
plt.show()

