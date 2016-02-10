#!/usr/bin/env python

### This code is co-authored by Zoltan Onodi-Szucs, Palash Matey and Maanit Mehra with
### non-original portions based on: https://github.com/Itseez/opencv/blob/master/samples/python2/stereo_match.py

'''
The code below implements a basic stereo match and follows that up with a point cloud generation (.ply file).
Resulting .ply file cam be easily viewed using MeshLab ( http://meshlab.sourceforge.net/ )

While the below code can be used for any image in general, using the openCV optimizations, 
the code can be modified to work with a disparity map generated using any of the alternate optimizations attached
in the optimization files. Note however, the latter approach needs to be customised to a generic case. The openCV
libraries allow for that by default.

Our project focused on the optimization of the matching algorithm, we do NOT focus on the point cloud optimization.
'''

import numpy as np
import cv2

ply_header = '''ply
format ascii 1.0
element vertex %(vert_num)d
property float x
property float y
property float z
property uchar red
property uchar green
property uchar blue
end_header
'''

def write_ply(fn, verts, colors):
    verts = verts.reshape(-1, 3)
    colors = colors.reshape(-1, 3)
    verts = np.hstack([verts, colors])
    with open(fn, 'w') as f:
        f.write(ply_header % dict(vert_num=len(verts)))
        np.savetxt(f, verts, '%f %f %f %d %d %d')


if __name__ == '__main__':
    print 'loading images...'
    imgL = cv2.imread('left.png')  
    imgR = cv2.imread('right.png')

#############################################################
#################### CODE ALGORITHM BELOW ###################
#############################################################


    
    window_size = 3		# NOTE: must be in the 3..11 range
    min_disp = 64
    num_disp = 112-min_disp	# numDisparities is typically maxDisparity - minDisparity


    #cv2.StereoSGBM([minDisparity, numDisparities, SADWindowSize[, P1[, P2[, disp12MaxDiff[, preFilterCap[, uniquenessRatio[, speckleWindowSize[, speckleRange[, fullDP]]]]]]]]])
    stereo = cv2.StereoSGBM(minDisparity = min_disp, numDisparities=num_disp, SADWindowSize=window_size, P1 = 8*3*window_size**2, P2 = 32*3*window_size**2, disp12MaxDiff = 1, uniquenessRatio = 10, speckleWindowSize = 100, speckleRange = 32)
    #http://docs.opencv.org/2.4/modules/calib3d/doc/camera_calibration_and_3d_reconstruction.html#id5 for a full description of all the parameters
	
    
##############################################################
############## BELOW SECTION CAN BE DONE AS IT IS ############
##############################################################
    print 'generating the disparity map'
    disp = stereo.compute(imgL, imgR).astype(np.float32) / 16.0

    print 'generating 3d point cloud...',
    h, w = imgL.shape[:2]
    f = 0.8*w                          # an assumption focal length

#### Q represents the transform matrix from one plane to the other ####
#### Q here represents the [x y z 0]T vector ####

    Q = np.float32([[1, 0, 0, -0.5*w],
                    [0,-1, 0,  0.5*h], # turn points 180 deg around x-axis,
                    [0, 0, 0,     -f], # so that y-axis looks up
                    [0, 0, 1,      0]])

### COnverting to pointcloud ###
    points = cv2.reprojectImageTo3D(disp, Q)

### Colored Point Cloud ###
    colors = cv2.cvtColor(imgL, cv2.COLOR_BGR2RGB)
    mask = disp > disp.min()
    out_points = points[mask]
    out_colors = colors[mask]
    out_fn = 'out.ply'
    write_ply('out.ply', out_points, out_colors)
    print '%s saved' % 'out.ply'

## Uncomment these lines when working on your system ###
#    cv2.imshow('left', imgL)
#    cv2.imshow('disparity', (disp-min_disp)/num_disp)
    cv2.waitKey()
    cv2.destroyAllWindows()
