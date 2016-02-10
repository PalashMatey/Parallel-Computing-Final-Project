import cv2
import time
import numpy as np
import pyopencl as cl
import matplotlib as mpl
mpl.use('agg')
import matplotlib.pyplot as plt
mpl.rcParams['savefig.dpi'] = 100
from pylab import *
import csv
import pyopencl.array


# Select the desired OpenCL platform; you shouldn't need to change this:
NAME = 'NVIDIA CUDA'
platforms = cl.get_platforms()
devs = None
for platform in platforms:
    if platform.name == NAME:
        devs = platform.get_devices()

print devs

# Set up a command queue:
ctx = cl.Context(devs)
queue = cl.CommandQueue(ctx)

###########################################################
######## Setting up the OpenCL environment ################
###########################################################

left = cv2.imread('left.png',0).astype(np.float32)
right = cv2.imread('right.png',0).astype(np.float32)

#print left
print left.shape, " is the image dimension."

disparityRange = 64
halfBlockSize = 3
blockSize = (2 * halfBlockSize + 1)
DbasicSubpixel = np.zeros(left.shape).astype(np.float32)
imgHeight, imgWidth = left.shape

#print imgWidth, "This is image Width"

###########################################################
####################### KERNEL 1 ##########################
###########################################################

func1 = cl.Program(ctx, """
#include <pyopencl-complex.h>
__kernel void basic1(const int halfBlockSize, const int blockSize, const int disparityRange ,__global float* left, __global float* right, const int imgHeight , const int imgWidth, __global float* output) {
	
	int row = get_global_id(1);
	int col = get_global_id(0);
	const int elements = (2*halfBlockSize+1)*(2*halfBlockSize+1);
        //int* template = calloc(elements, sizeof(int));
        //int* block = calloc(elements, sizeof(int));
	//__local int template[49];
	//__local int block[49];
	int template[49];
	int block[49];
	int l,i,j,t;
	int minc,maxc,mind,maxd,numBlocks,elemin,elementnum,blockIndex;
	int blockDiffs[65];
	float d;
	int C1,C2,C3,bestMatchIndex ;  
	int h,k,r,sum1,indx;
	
	int minr = 0 > (row - halfBlockSize) ? 0 : (row - halfBlockSize); 
	int maxr = (imgHeight - 1) < (row + halfBlockSize) ? (imgHeight - 1) : (row + halfBlockSize) ;
	
	minc = (0 > (col - halfBlockSize) ? 0 : (col - halfBlockSize)); 
	maxc = (imgWidth - 1) < (col + halfBlockSize) ? (imgWidth - 1) : (col + halfBlockSize);

	mind = 0;
	maxd = disparityRange < (imgWidth - 1 - maxc) ? disparityRange : (imgWidth - 1 - maxc); 

	elementnum = (maxr+1-minr)*(maxc+1-minc);
	int templateWidth = maxc+1-minc;  

        //////////////////template/////////////////////
	h = 0;
        for ( i = minr ; i <= maxr ; i++ )
	{		
		k = 0;
		for (j = minc ; j <= maxc ; j++)
		{
		        template[k + h * templateWidth] = right[i * imgWidth + j];
                        k++;
		} 
	        h++;
	}
        ///////////////////////////////////////////////
        
	numBlocks = maxd - mind + 1;
	//blockDiffs[numBlocks-1]={255};
		 
	for ( j = mind ; j <= maxd ; j++)
	{
		h = 0;
		for ( k = minr ; k <= maxr ; k++)
		{
			r = 0;
			for (l = minc+j ; l <= maxc+j ; l++)
			{
	             		block[r + h * templateWidth] = left[k * imgWidth + l];
				r++;
			}
			h++;
		}	
		blockIndex = j;
		sum1 = 0;
		for ( r = 0 ; r < elementnum ; r ++ )
		{
			sum1 = sum1 + abs(template[r]- block[r]);
		}  
		blockDiffs[blockIndex] = sum1;
	}
	elemin = blockDiffs[0];
	indx = 0;
	for ( t = 0 ; t < numBlocks ; t++)
	{
		if (blockDiffs[t] < elemin )
		{
			indx = t ; 
			elemin = blockDiffs[t];
		}
	}			
	bestMatchIndex = indx;
	d = bestMatchIndex ;
	//output[row*imgWidth + col]= row*imgWidth + col;
	if (bestMatchIndex == 0 || bestMatchIndex == numBlocks - 1)
	{
		output[row*imgWidth + col] = d;
	}
	else
	{
		C1 = blockDiffs[bestMatchIndex - 1];
		C2 = blockDiffs[bestMatchIndex ];
		C3 = blockDiffs[bestMatchIndex + 1];
		output[row* imgWidth + col] = d - (0.5*(C3-C1)/(C1 - 2*C2 + C3));
	}
	//free(template);
	//free(block);
}
""").build().basic1
""" """
func1.set_scalar_arg_dtypes([np.uint32, np.uint32, np.uint32,  None, None, np.uint32, np.uint32, None])

def hist_op_1_time(BS, D, left_buf, right_buf, H, W, out_buf):
	start = time.time()
	func1(queue, (H,W), None, np.int32(BS/2) ,np.int32(BS) , np.int32(D) , left_buf, right_buf,np.int32(H), np.int32(W), out_buf)
	return time.time()-start


def hist_op_1_data (BS, D, left, right, H, W, out):                                                                                                               
        left_buf,right_buf, out_buf = mem_alloc(left,right,out)
        t=hist_op_1_time(BS, D, left_buf, right_buf, H, W, out_buf)
        out=mem_transfer(out,out_buf)
        return t, out

##########################################################################

func2 = cl.Program(ctx, """
#include <pyopencl-complex.h>
__kernel void basic2(const int halfBlockSize, const int blockSize, const int disparityRange ,__global float* left, __global float* right, const int imgHeight , const int imgWidth, __global float* output2) {
	
	int row = get_global_id(1);
	int col = get_global_id(0);
	const int elements = (2*halfBlockSize+1)*(2*halfBlockSize+1);
        //int* template = calloc(elements, sizeof(int));
        //int* block = calloc(elements, sizeof(int));
	__local int template[49];
	__local int block[49];
	int l,i,j,t;
	int minc,maxc,mind,maxd,numBlocks,elemin,elementnum,blockIndex;
	int blockDiffs[65];
	float d;
	int C1,C2,C3,bestMatchIndex ;  
	int h,k,r,sum1,indx;
	
	int minr = 0 > (row - halfBlockSize) ? 0 : (row - halfBlockSize); 
	int maxr = (imgHeight - 1) < (row + halfBlockSize) ? (imgHeight - 1) : (row + halfBlockSize) ;
	
	minc = (0 > (col - halfBlockSize) ? 0 : (col - halfBlockSize)); 
	maxc = (imgWidth - 1) < (col + halfBlockSize) ? (imgWidth - 1) : (col + halfBlockSize);

	mind = 0;
	maxd = disparityRange < (imgWidth - 1 - maxc) ? disparityRange : (imgWidth - 1 - maxc); 

	elementnum = (maxr+1-minr)*(maxc+1-minc);
	int templateWidth = maxc+1-minc;  

        //////////////////template/////////////////////
	h = 0;
        for ( i = minr ; i <= maxr ; i++ )
	{		
		k = 0;
		for (j = minc ; j <= maxc ; j++)
		{
		        template[k + h * templateWidth] = right[i * imgWidth + j];
                        k++;
		} 
	        h++;
	}
	barrier(CLK_LOCAL_MEM_FENCE);
        ///////////////////////////////////////////////
        
	numBlocks = maxd - mind + 1;
	//blockDiffs[numBlocks-1]={255};
		 
	for ( j = mind ; j <= maxd ; j++)
	{
		h = 0;
		for ( k = minr ; k <= maxr ; k++)
		{
			r = 0;
			for (l = minc+j ; l <= maxc+j ; l++)
			{
	             		block[r + h * templateWidth] = left[k * imgWidth + l];
				r++;
			}
			h++;
		
		}
		barrier(CLK_LOCAL_MEM_FENCE);	
		blockIndex = j;
		sum1 = 0;
		for ( r = 0 ; r < elementnum ; r ++ )
		{
			sum1 = sum1 + abs(template[r]- block[r]);
		}  
		blockDiffs[blockIndex] = sum1;
	}
	elemin = blockDiffs[0];
	indx = 0;
	for ( t = 0 ; t < numBlocks ; t++)
	{
		if (blockDiffs[t] < elemin )
		{
			indx = t ; 
			elemin = blockDiffs[t];
		}
	}			
	bestMatchIndex = indx;
	d = bestMatchIndex ;
	//output[row*imgWidth + col]= row*imgWidth + col;
	if (bestMatchIndex == 0 || bestMatchIndex == numBlocks - 1)
	{
		output2[row*imgWidth + col] = d;
	}
	else
	{
		C1 = blockDiffs[bestMatchIndex - 1];
		C2 = blockDiffs[bestMatchIndex ];
		C3 = blockDiffs[bestMatchIndex + 1];
		output2[row* imgWidth + col] = d - (0.5*(C3-C1)/(C1 - 2*C2 + C3));
	}
	//free(template);
	//free(block);
}
""").build().basic2
""" """

func2.set_scalar_arg_dtypes([np.uint32, np.uint32, np.uint32,  None, None, np.uint32, np.uint32, None])


def hist_op_2_time(BS, D, left_buf, right_buf, H, W, out_buf):
    start = time.time()
    func2(queue, (H,W), None, np.int32(BS/2) ,np.int32(BS) , np.int32(D) , left_buf, right_buf,np.int32(H), np.int32(W), out_buf)
    return time.time()-start


def hist_op_2_data(BS,D, left, right, H, W, out):                                                                                             
        left_buf,right_buf, out_buf = mem_alloc(left,right,out)
        t=hist_op_2_time(BS, D, left_buf, right_buf, H, W, out_buf)
        out=mem_transfer(out,out_buf)
        return t, out


###################################################################
################# KERNEL 3 #######################################
##################################################################
func3 = cl.Program(ctx, """
#include <pyopencl-complex.h>
__kernel void basic3(const int halfBlockSize, const int blockSize, const int disparityRange ,__global float* left, __global float* right, const int imgHeight , const int imgWidth, __global float* output3) {

        int row = get_global_id(1);
        int col = get_global_id(0);
        const int elements = (2*halfBlockSize+1)*(2*halfBlockSize+1);
        //int* template = calloc(elements, sizeof(int));
        //int* block = calloc(elements, sizeof(int));
        private int template[49];
        private int block[49];
        int l,i,j,t;
        int minc,maxc,mind,maxd,numBlocks,elemin,elementnum,blockIndex;
        int blockDiffs[65];
        float d;
        int C1,C2,C3,bestMatchIndex ;
        int h,k,r,sum1,indx;

        int minr = 0 > (row - halfBlockSize) ? 0 : (row - halfBlockSize);
        int maxr = (imgHeight - 1) < (row + halfBlockSize) ? (imgHeight - 1) : (row + halfBlockSize) ;

        minc = (0 > (col - halfBlockSize) ? 0 : (col - halfBlockSize));
        maxc = (imgWidth - 1) < (col + halfBlockSize) ? (imgWidth - 1) : (col + halfBlockSize);

        mind = 0;
        maxd = disparityRange < (imgWidth - 1 - maxc) ? disparityRange : (imgWidth - 1 - maxc);

        elementnum = (maxr+1-minr)*(maxc+1-minc);
        int templateWidth = maxc+1-minc;

        //////////////////template/////////////////////
        h = 0;
        for ( i = minr ; i <= maxr ; i++ )
        {               
                k = 0;
                for (j = minc ; j <= maxc ; j++)
                {
                        template[k + h * templateWidth] = right[i * imgWidth + j];
                        k++;
                }
                h++;
        }
        ///////////////////////////////////////////////

	numBlocks = maxd - mind + 1;
        //blockDiffs[numBlocks-1]={255};

        for ( j = mind ; j <= maxd ; j++)
        {
                h = 0;
                for ( k = minr ; k <= maxr ; k++)
                {
                        r = 0;
                        for (l = minc+j ; l <= maxc+j ; l++)
                        {
                                block[r + h * templateWidth] = left[k * imgWidth + l];
                                r++;
                        }
                        h++;
                }
                blockIndex = j;
                sum1 = 0;
                for ( r = 0 ; r < elementnum ; r ++ )
                {
                        sum1 = sum1 + abs(template[r]- block[r]);
                }
                blockDiffs[blockIndex] = sum1;
        }
        elemin = blockDiffs[0];
	 indx = 0;
        for ( t = 0 ; t < numBlocks ; t++)
        {
                if (blockDiffs[t] < elemin )
                {
                        indx = t ;
                        elemin = blockDiffs[t];
                }
        }
        bestMatchIndex = indx;
        d = bestMatchIndex ;
        //output[row*imgWidth + col]= row*imgWidth + col;
        if (bestMatchIndex == 0 || bestMatchIndex == numBlocks - 1)
        {
                output3[row*imgWidth + col] = d;
        }
        else
        {
                C1 = blockDiffs[bestMatchIndex - 1];
                C2 = blockDiffs[bestMatchIndex ];
                C3 = blockDiffs[bestMatchIndex + 1];
                output3[row* imgWidth + col] = d - (0.5*(C3-C1)/(C1 - 2*C2 + C3));
        }
        //free(template);
        //free(block);
}
""").build().basic3
""" """
func3.set_scalar_arg_dtypes([np.uint32, np.uint32, np.uint32,  None, None, np.uint32, np.uint32, None])

def hist_op_3_time(BS, D, left_buf, right_buf, H, W, out_buf):
    start = time.time()
    func3(queue, (H,W), None, np.int32(BS/2) ,np.int32(BS) , np.int32(D) , left_buf, right_buf,np.int32(H), np.int32(W), out_buf)
    return time.time()-start


def hist_op_3_data(BS,D, left, right, H, W, out):                                                                                             
        left_buf,right_buf, out_buf = mem_alloc(left,right,out)
        t=hist_op_3_time(BS, D, left_buf, right_buf, H, W, out_buf)
        out=mem_transfer(out,out_buf)
        return t, out

###################################################################
################# KERNEL 4 #######################################
##################################################################

func4=cl.Program(ctx, """
#include <pyopencl-complex.h>
__kernel void basic4(__local int* rightBlock, __local int* leftBlock, const int halfBlockSize, const int disparityRange ,__global float* left, __global float* right, const int imgHeight, const int imgWidth, __global float* output) {
	
	int row = get_global_id(0);
	int col = get_global_id(1);

        int loc_row = get_local_id(0);
        int loc_col = get_local_id(1);

        int grp_id_x = get_group_id(0);
        int grp_id_y = get_group_id(1);


	int offset = halfBlockSize*imgWidth+halfBlockSize;
        int blockSize = 2*halfBlockSize + 1;
	int idx = loc_col+loc_row*9;

	volatile __local int disp_min;
	volatile __local int bin[63];

	if (loc_row < 7 && loc_col < 7)
	{
		rightBlock[loc_col+loc_row*blockSize] = right[loc_col+loc_row*imgWidth + grp_id_y+grp_id_x*imgWidth];
	}
	barrier(CLK_LOCAL_MEM_FENCE);
	
	int priv_rightBlock[49];  //7x7

	for (int k = 0; k < blockSize*blockSize; k++)
        {
                priv_rightBlock[k] = rightBlock[k];
        }




	int loop_length = 8;
	if (loc_col > 7)
	{
		loop_length = 5;
	}
	
	for (int i = loc_col*8; i < loc_col*8+loop_length; i++)
	{
		leftBlock[i+loc_row*69] = left[i+loc_row*imgWidth + grp_id_y+grp_id_x*imgWidth];
	}
	barrier(CLK_LOCAL_MEM_FENCE);
	
	int priv_leftBlock[49];   //7x7
	
	for (int n = 0; n < 7; n++)
	{
		for (int m = 0; m < 7; m++)
		{
			priv_leftBlock[m+n*7] = leftBlock[m+idx+n*69];
		}
	} 

	

	
	int sum = 0;
	for (int l = 0; l < 49; l++)
	{
		sum = sum + abs(priv_leftBlock[l] - priv_rightBlock[l]); 
	}

	bin[idx] = sum;
	barrier(CLK_LOCAL_MEM_FENCE);

	int elemin = 1024;
	if(idx == 1)
	{
		disp_min = 0;
		for (int z = 0; z<63; z++) 
		{
			if (bin[z] < elemin )
               	 	{
                	       	disp_min = z;
                      		elemin = bin[z];
                	}
		}
        	output[grp_id_y + grp_id_x*(imgWidth-6-disparityRange)] = disp_min;
	}
	barrier(CLK_LOCAL_MEM_FENCE);
}

""").build().basic4
""" """
func4.set_scalar_arg_dtypes([None, None, np.uint32, np.uint32,  None, None, np.uint32, np.uint32, None])

def hist_op_4_time(BS, D, left_buf, right_buf, H, W, out_buf):
    start = time.time()
    global_size = np.zeros(shape=(7*(H-6),9*(W-6-D))).astype(np.float32) # 7x9 group shape 63 work item
    group_size = np.zeros(shape=(7,9)).astype(np.float32)
    lclLeft = cl.LocalMemory(np.int32().nbytes*483)
    lclRight = cl.LocalMemory(np.int32().nbytes*49)
    func4(queue, global_size.shape, group_size.shape, lclRight, lclLeft, np.int32(BS/2), np.int32(D), left_buf, right_buf, np.int32(H), np.int32(W), out_buf)
    return time.time()-start


def hist_op_4_data(BS,D, left, right, H, W, out):
        left_buf,right_buf, out_buf = mem_alloc(left,right,out)
        t=hist_op_3_time(BS, D, left_buf, right_buf, H, W, out_buf)
        out=mem_transfer(out,out_buf)
        return t, out


###################################################################
################# KERNEL 5 #######################################
##################################################################

func5=cl.Program(ctx, """
#include <pyopencl-complex.h>
__kernel void basic5(__local int* rightBlock, __local int* leftBlock, const int halfBlockSize, const int disparityRange ,__global float* left, __global float* right, const int imgHeight, const int imgWidth, __global float* output) {
	
	int row = get_global_id(0);
	int col = get_global_id(1);

        int loc_row = get_local_id(0);
        int loc_col = get_local_id(1);

        int grp_id_x = get_group_id(0);
        int grp_id_y = get_group_id(1);


	int offset = halfBlockSize*imgWidth+halfBlockSize;
        int blockSize = 2*halfBlockSize + 1;
	int idx = loc_col+loc_row*9;

	volatile __local int disp_min;
	volatile __local int bin[64];
	volatile __local int location[64];
	volatile __local int factor;

	if (loc_row < 7 && loc_col < 7)
	{
		rightBlock[loc_col+loc_row*blockSize] = right[loc_col+loc_row*imgWidth + grp_id_y+grp_id_x*imgWidth];
	}




	int loop_length = 8;
	if (loc_col > 7)
	{
		loop_length = 5;
	}
	
	for (int i = loc_col*8; i < loc_col*8+loop_length; i++)
	{
		leftBlock[i+loc_row*69] = left[i+loc_row*imgWidth + grp_id_y+grp_id_x*imgWidth];
	}
	barrier(CLK_LOCAL_MEM_FENCE);
	

	
	int sum = 0;
	for (int n = 0; n < 7; n++)
	{
		for (int m = 0; m < 7; m++)
		{
			sum = sum + abs(leftBlock[m+idx+n*69] - rightBlock[m+n*7]); 
		}
	}
	bin[idx] = sum;
	barrier(CLK_LOCAL_MEM_FENCE);

	//int elemin = 1024;
	//if(idx == 1)
	//{
	//	disp_min = 0;
	//	for (int z = 0; z<63; z++) 
	//	{
	//		if (bin[z] < elemin )
        //      	 	{
        //        	       	disp_min = z;
        //              		elemin = bin[z];
        //       	}
	//	}
        //	output[grp_id_y + grp_id_x*(imgWidth-6-disparityRange)] = disp_min;
	//}
	//barrier(CLK_LOCAL_MEM_FENCE);


	location[idx] = idx;
	if(idx == 0)
        {
	        factor = 2;
		location[63] = 63;
		bin[63] = 1024;
        }

	int element = 64;
	while (element/factor !=0)
	{
		if(idx/(element/factor) == 0)
		{
			location[idx] = (bin[idx*2]<bin[idx*2+1]) ? location[idx*2] : location[idx*2+1];
			bin[idx] = (bin[idx*2]<bin[idx*2+1]) ? bin[idx*2] : bin[idx*2+1];
		}		
		barrier(CLK_LOCAL_MEM_FENCE);
		if(idx == 0)
		{
			factor = factor*2;
		}
		barrier(CLK_LOCAL_MEM_FENCE);	
	}

	if(idx == 0)
        {
		output[grp_id_y + grp_id_x*(imgWidth-6-disparityRange)] = location[idx];
	}


}
""").build().basic5
""" """

func5.set_scalar_arg_dtypes([None, None, np.uint32, np.uint32,  None, None, np.uint32, np.uint32, None])

def hist_op_5_time(BS, D, left_buf, right_buf, H, W, out_buf):
    start = time.time()
    global_size = np.zeros(shape=(7*(H-6),9*(W-6-D))).astype(np.float32) # 7x9 group shape 63 work item
    group_size = np.zeros(shape=(7,9)).astype(np.float32)
    lclLeft = cl.LocalMemory(np.int32().nbytes*483)
    lclRight = cl.LocalMemory(np.int32().nbytes*49)
    func5(queue, global_size.shape, group_size.shape, lclRight, lclLeft, np.int32(BS/2), np.int32(D), left_buf, right_buf, np.int32(H), np.int32(W), out_buf)
    return time.time()-start


def hist_op_5_data(BS,D, left, right, H, W, out):
        left_buf,right_buf, out_buf = mem_alloc(left,right,out)
        t=hist_op_5_time(BS, D, left_buf, right_buf, H, W, out_buf)
        out=mem_transfer(out,out_buf)
        return t, out




###################################################################
################# KERNEL 6 #######################################
##################################################################

func6=cl.Program(ctx, """
#include <pyopencl-complex.h>
__kernel void basic6(__local int* rightBlock, __local int* leftBlock, const int halfBlockSize, const int disparityRange ,__global float* left, __global float* right, const int imgHeight, const int imgWidth, __global float* output) {
	
	int row = get_global_id(0);
	int col = get_global_id(1);

        //int loc_row = get_local_id(0);
        int loc_col = get_local_id(1);

        int grp_id_x = get_group_id(0);
        int grp_id_y = get_group_id(1);


	int offset = halfBlockSize*imgWidth+halfBlockSize;
        int blockSize = 2*halfBlockSize + 1;
	//int idx = loc_col+loc_row*9;
	int SAD[65];
	int idx[65];

	//volatile __local int disp_min;
	//volatile __local int bin[63];



	//if (loc_row < 7 && loc_col < 7)
	//{
	//	rightBlock[loc_col+loc_row*blockSize] = right[loc_col+loc_row*imgWidth + grp_id_y+grp_id_x*imgWidth];
	//}

	if (loc_col < 14)
	{
		for (int h = loc_col*13; h < loc_col*13+13; h++)
		{
			rightBlock[h] = right[h%26+(h/26)*imgWidth+grp_id_y*20+grp_id_x*imgWidth];
		}
	}

	


	int loop_length = 30;
	if (loc_col/18==1)
	{
		loop_length = 45;
	}
	for (int i = loc_col*30 + (loc_col/19)*15; i < loc_col*30 + (loc_col/19)*15+loop_length; i++)
	{
		leftBlock[i] = left[i%90 + (i/90)*imgWidth + grp_id_y*20 + grp_id_x*imgWidth];
	}
	barrier(CLK_LOCAL_MEM_FENCE);
	



        int priv_rightBlock[49];  //7x7
        for (int k = 0; k < blockSize*blockSize; k++)
        {
                priv_rightBlock[k] = rightBlock[k%7 + (k/7)*26 + loc_col];
        }



	int priv_leftBlock[49];
	int sum;
	for (int d=0; d < disparityRange; d++)
	{
		for (int m = 0; m < 49; m++)
		{
			priv_leftBlock[m] = leftBlock[m%7 + (m/7)*90 + d + loc_col];
		}
		
		sum = 0;
		for (int j = 0; j<49; j++)
		{
			sum = sum + abs(priv_rightBlock[j]-priv_leftBlock[j]);		
		}
		SAD[d] = sum;
	}	
	
	int elemin = SAD[0];
	int index = 0;
	for (int l = 1; l<disparityRange; l++)
	{
		if(SAD[l] < elemin)
		{
			index = l;
			elemin = SAD[l];	
		}	

	}
	//output[loc_col+grp_id_y*20+grp_id_x*imgWidth]=index;
	output[col + row*380]=index;
}



""").build().basic6


func6.set_scalar_arg_dtypes([None, None, np.uint32, np.uint32,  None, None, np.uint32, np.uint32, None])

def hist_op_6_time(BS, D, left_buf, right_buf, H, W, out_buf):
    start = time.time()
    global_size = np.zeros(shape=(7*(H-6),9*(W-6-D))).astype(np.float32) # 7x9 group shape 63 work item
    group_size = np.zeros(shape=(7,9)).astype(np.float32)
    lclLeft = cl.LocalMemory(np.int32().nbytes*630)
    lclRight = cl.LocalMemory(np.int32().nbytes*182)
    func6(queue, global_size.shape, group_size.shape, lclRight, lclLeft, np.int32(BS/2), np.int32(D), left_buf, right_buf, np.int32(H), np.int32(W), out_buf)
    return time.time()-start


def hist_op_6_data(BS,D, left, right, H, W, out):
        left_buf,right_buf, out_buf = mem_alloc(left,right,out)
        t=hist_op_6_time(BS, D, left_buf, right_buf, H, W, out_buf)
        out=mem_transfer(out,out_buf)
        return t, out




###################################################################################################################################



#########################################################################################
#########################################################################################
#########################################################################################

def create_arrays(H,W):
        #N=R*C
        #img = np.random.randint(0, 255, N).astype(np.uint8).reshape(R*C)
        #img = np.memmap('/opt/data/random.dat', dtype=np.uint8, mode='r').reshape(R*C,)
        left = np.memmap('/opt/data/random.dat', dtype=np.uint8, mode='r',shape=(H,W))
        right = np.memmap('/opt/data/random.dat', dtype=np.uint8, mode='r',shape=(H,W))
	out=np.zeros(left.shape).astype(np.float32)
        return left,right,out



def mem_alloc(left, right, out):
	mf = cl.mem_flags
	left_buf = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=left)
	right_buf = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=right)
	#r_buf = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=readimg)  
	out_buf = cl.Buffer(ctx, mf.WRITE_ONLY | mf.COPY_HOST_PTR, hostbuf=out)
        init_arr=np.zeros(out.shape).astype(np.float32)
        cl.enqueue_copy(queue,out_buf,init_arr)                           #Initializing the Memory of the Output Buffers 
        return left_buf, right_buf, out_buf



#prg=cl.Program(ctx,kernel).build()                                     #PROGRAM
#prg.mat_transpose(queue,A_trans.shape,None,a_buf,atrans_buf,np.uint32(height_A),np.uint32(width_A))    #KERNEL LAUNCH
def mem_transfer(out, out_buf):
        cl.enqueue_copy(queue,out,out_buf)                                          #Copying Final Data into Python Buffers
        return out



#########################################################################################
#########################################################################################
#########################################################################################

def python_basic(BS,D, left, right, H, W, out):
	imgHeight=H
	imgWidth=W
	disparityRange=D
	halfBlockSize=int(BS/2)

	start_time=time.time()
	for m in np.arange(0 , imgHeight):

		minr = np.maximum(0, m - halfBlockSize)
  		maxr = np.minimum(imgHeight-1, m + halfBlockSize)

    		for n in range (0, imgWidth):

        		minc = np.maximum(0, n - halfBlockSize)
        		maxc = np.minimum(imgWidth-1, n + halfBlockSize)

        		mind = 0
        		maxd = np.minimum(disparityRange, imgWidth - maxc)
        		template = np.array(right[minr:maxr+1, minc:maxc+1])
        		numBlocks = maxd - mind + 1

		        blockDiffs = np.zeros((numBlocks-1, 1))

	        	for i in range (mind, maxd):

                		block = np.array(left[minr:maxr+1, minc + i: maxc + i+1])

		                blockIndex = i - mind +1 -1

                		blockDiffs[blockIndex, 0] = np.sum(np.sum(np.absolute(template - block)))
                		z=blockDiffs.shape

 
        		sortedIndeces = np.argmin(blockDiffs,axis = 0);
        		if type(sortedIndeces)==np.ndarray:
                		bestMatchIndex = sortedIndeces[0]
        		else:
		                bestMatchIndex = sortedIndeces

        		d = (bestMatchIndex) + mind
        		if ((bestMatchIndex == 0 ) or (bestMatchIndex == numBlocks-2)):
		                DbasicSubpixel[m, n] = d
                		ctr1=ctr1+1;
        		else :
                		C1 = blockDiffs[bestMatchIndex -1]
                		C2 = blockDiffs[bestMatchIndex]
                		C3 = blockDiffs[bestMatchIndex + 1]
                		DbasicSubpixel[m, n] = d - (0.5 * (C3 - C1) / (C1 - (2*C2) + C3))

#		print type(DbasicSubpixel), ": Type Dbasicsubpixel"
#		print DbasicSubpixel.shape, ": its shape"
	t=time.time()-start
	minDBS=np.min(DbasicSubpixel)
	maxDBS=np.max(DbasicSubpixel)

	DbasicSubpixel=150*((DbasicSubpixel-minDBS)/(maxDBS-minDBS))
#		print DbasicSubpixel
#		cv2.imwrite('disparity_with_Sobel-150.png',DbasicSubpixel)
#		print "ctr1=%d,  ctr2=%d" %(ctr1,ctr2)

#		np.savetxt('DbasicSubpixel.xls',DbasicSubpixel,delimiter='\t')
	out = DbasicSubpixel

	return t, out

######################################################################################################################################################################

##### Evaluating Array Timings: ########


#Only in the case of python execution do we keep the M=1
def py_hist_time(BS,D,H,W,M=1):
        times = []
        left,right,out =create_arrays(H,W)
        left_buf, right_buf, out_buf = mem_alloc(left,right, out)
        for i in xrange(M):
                t,y=python_basic(BS, D, left, right, H, W, out)
                times.append(t)
        #print 'python time:  ', np.average(times)
        return np.average(times)

def cl_op1_hist_time(BS,D,H,W, M=4):
        times = []
        left,right, out =create_arrays(H,W)
        left_buf,right_buf, out_buf = mem_alloc(left,right, out)
        for i in xrange(M):
                #t=hist_op_1_time(img_buf,bins_buf, N)
		t=hist_op_1_time(BS, D, left_buf, right_buf, H, W, out_buf)
                times.append(t)
                out=mem_transfer(out,out_buf)
        #print 'opencl time:  ', np.average(times)
        return np.average(times)


def cl_op2_hist_time(BS,D,H,W, M=4):
        times = []
        left,right, out =create_arrays(H,W)
        left_buf,right_buf, out_buf = mem_alloc(left,right, out)
        for i in xrange(M):
                #t=hist_op_1_time(img_buf,bins_buf, N)
                t=hist_op_2_time(BS, D, left_buf, right_buf, H, W, out_buf)
                times.append(t)
                out=mem_transfer(out,out_buf)
        #print 'opencl time:  ', np.average(times)
        return np.average(times)


def cl_op3_hist_time(BS,D,H,W, M=4):
        times = []
        left,right, out =create_arrays(H,W)
        left_buf,right_buf, out_buf = mem_alloc(left,right, out)
        for i in xrange(M):
                #t=hist_op_1_time(img_buf,bins_buf, N)
                t=hist_op_3_time(BS, D, left_buf, right_buf, H, W, out_buf)
                times.append(t)
                out=mem_transfer(out,out_buf)
        #print 'opencl time:  ', np.average(times)
        return np.average(times)


def cl_op4_hist_time(BS,D,H,W, M=4):
        times = []
        left,right, out =create_arrays(H,W)
        left_buf,right_buf, out_buf = mem_alloc(left,right, out)
        for i in xrange(M):
                #t=hist_op_1_time(img_buf,bins_buf, N)
                t=hist_op_4_time(BS, D, left_buf, right_buf, H, W, out_buf)
                times.append(t)
                out=mem_transfer(out,out_buf)
        #print 'opencl time:  ', np.average(times)
        return np.average(times)


def cl_op5_hist_time(BS,D,H,W, M=4):
        times = []
        left,right, out =create_arrays(H,W)
        left_buf,right_buf, out_buf = mem_alloc(left,right, out)
        for i in xrange(M):
                #t=hist_op_1_time(img_buf,bins_buf, N)
                t=hist_op_5_time(BS, D, left_buf, right_buf, H, W, out_buf)
                times.append(t)
                out=mem_transfer(out,out_buf)
        #print 'opencl time:  ', np.average(times)
        return np.average(times)

def cl_op6_hist_time(BS,D,H,W, M=4):
        times = []
        left,right, out =create_arrays(H,W)
        left_buf,right_buf, out_buf = mem_alloc(left,right, out)
        for i in xrange(M):
                #t=hist_op_1_time(img_buf,bins_buf, N)
                t=hist_op_6_time(BS, D, left_buf, right_buf, H, W, out_buf)
                times.append(t)
                out=mem_transfer(out,out_buf)
        #print 'opencl time:  ', np.average(times)
        return np.average(times)


####### BASIC OPENCV #######
def cl_op7_hist_time(BS,D,H,W,M=4):
	times = []
        left,right, out =create_arrays(H,W)
        for i in xrange(M):
                #t=hist_op_1_time(img_buf,bins_buf, N)
                start=time.time()
		stereo = cv2.StereoBM(cv2.STEREO_BM_BASIC_PRESET,ndisparities=D, SADWindowSize=BS)
        	##http://docs.opencv.org/2.4.1/modules/calib3d/doc/camera_calibration_and_3d_reconstruction.html#stereobm-stereobm
        	#out = stereo.compute(left,right)
                t=time.time()-start
		times.append(t)
        #print 'opencl time:  ', np.average(times)
        return np.average(times)


H=320
W=480
D=16
BS=7


##### PLot against a Variation in BS ####
python_times=[]
pyopencl_op1_times=[]
pyopencl_op2_times=[]
pyopencl_op3_times=[]
pyopencl_op4_times=[]
pyopencl_op5_times=[]
pyopencl_op6_times=[]
pyopencl_op7_times=[]

param=np.arange(3,8,2).astype(np.int32)

for i in param:
	pyopencl_op1_times.append(cl_op1_hist_time(i,D,H,W,4))
	pyopencl_op2_times.append(cl_op2_hist_time(i,D,H,W,4))
        pyopencl_op3_times.append(cl_op3_hist_time(i,D,H,W,4))
        pyopencl_op4_times.append(cl_op4_hist_time(i,D,H,W,4))
        pyopencl_op5_times.append(cl_op5_hist_time(i,D,H,W,4))
        pyopencl_op6_times.append(cl_op6_hist_time(i,D,H,W,4))
        pyopencl_op7_times.append(cl_op7_hist_time(i,D,H,W,4))

plt.clf()
plt.plot(param, pyopencl_op1_times, 'r*-',
         param, pyopencl_op2_times, 'go-',
	 param, pyopencl_op3_times, 'bo-', 	
	 param, pyopencl_op7_times, 'mo-'
)

plt.xlabel('Num elements in Image')
plt.ylabel('$t$')
plt.title('Execution time as a function of Block Size (Basic optimizations)')
plt.legend(('OP1', 'OP2','OP3', 'OPCV'), loc='upper left')
plt.grid(True)
plt.gca().set_xlim((min(param), max(param)))
#plt.gca().set_ylim((0, 1.2*max(pyopencl_op1_times)))
#plt.draw()
plt.savefig('BS_scaling_basic.png')

plt.clf()
plt.plot(param, pyopencl_op4_times, 'ro-', 
         param, pyopencl_op5_times, 'g*-',
         param, pyopencl_op6_times, 'b*-',
)

plt.xlabel('Num elements in Image')
plt.ylabel('$t$')
plt.title('Execution time as a function of Block Size (Custom Memory optimizations)')
plt.legend(('OP4','OP5','OP6'), loc='upper left')
plt.grid(True)
plt.gca().set_xlim((min(param), max(param)))
#plt.gca().set_ylim((0, 1.2*max(pyopencl_op1_times)))
#plt.draw()
plt.savefig('BS_scaling_Mem_opti.png')

with open('BS_scaling.csv', 'w') as f:
    w = csv.writer(f)
    for a_size, t_op_1, t_op_2, t_op_3, t_op_4,t_op_5,t_op_6, t_op_7 in \
        zip(param,
            pyopencl_op1_times,
            pyopencl_op2_times,
            pyopencl_op3_times,
            pyopencl_op4_times,
	    pyopencl_op5_times,
	    pyopencl_op6_times,
            pyopencl_op7_times):
        w.writerow([a_size, t_op_1, t_op_2,t_op_3, t_op_4,t_op_5, t_op_6,t_op_7 ])


##### Plot against a Variation in H,W ####

python_times=[]
pyopencl_op1_times=[]
pyopencl_op2_times=[]
pyopencl_op3_times=[]
pyopencl_op4_times=[]
pyopencl_op5_times=[]
pyopencl_op6_times=[]
pyopencl_op7_times=[]

H=32
W=48

param=np.arange(1,13,3).astype(np.int32)

for i in param:
        pyopencl_op1_times.append(cl_op1_hist_time(BS,D,i*H,i*W,4))
        pyopencl_op2_times.append(cl_op2_hist_time(BS,D,i*H,i*W,4))
        pyopencl_op3_times.append(cl_op3_hist_time(BS,D,i*H,i*W,4))
        pyopencl_op4_times.append(cl_op4_hist_time(BS,D,i*H,i*W,4))
        pyopencl_op5_times.append(cl_op5_hist_time(BS,D,i*H,i*W,4))
        pyopencl_op6_times.append(cl_op6_hist_time(BS,D,i*H,i*W,4))
        pyopencl_op7_times.append(cl_op7_hist_time(BS,D,i*H,i*W,4))

plt.clf()
plt.plot(param, pyopencl_op1_times, 'r*-',
         param, pyopencl_op2_times, 'go-',
         param, pyopencl_op3_times, 'bo-',
         param, pyopencl_op7_times, 'mo-'
)

plt.xlabel('Num elements in Image')
plt.ylabel('$t$')
plt.title('Execution time as a function of Image Size (Basic optimizations)')
plt.legend(('OP1', 'OP2','OP3', 'OPCV'), loc='upper left')
plt.grid(True)
#plt.gca().set_xlim((min(param*H*param*W), max(param)))
#plt.gca().set_ylim((0, 1.2*max(pyopencl_op1_times)))
#plt.draw()
plt.savefig('Size_scaling_basic.png')

plt.clf()
plt.plot(param, pyopencl_op4_times, 'ro-',
         param, pyopencl_op5_times, 'g*-',
         param, pyopencl_op6_times, 'b*-',
)

plt.xlabel('Num elements in Image')
plt.ylabel('$t$')
plt.title('Execution time as a function of Image Size (Custom Memory optimizations)')
plt.legend(('OP4','OP5','OP6'), loc='upper left')
plt.grid(True)
#plt.gca().set_xlim((min(param), max(param)))
#plt.gca().set_ylim((0, 1.2*max(pyopencl_op1_times)))
#plt.draw()
plt.savefig('Size_scaling_Mem_opti.png')

with open('Size_scaling.csv', 'w') as f:
    w = csv.writer(f)
    w.writerow(['Size', 'op1', 'op2', 'op3', 'op4', 'op5', 'op6', 'op7'])
    for a_size, t_op_1, t_op_2, t_op_3, t_op_4,t_op_5,t_op_6, t_op_7 in \
        zip(param*param*H*W,
            pyopencl_op1_times,
            pyopencl_op2_times,
            pyopencl_op3_times,
            pyopencl_op4_times,
            pyopencl_op5_times,
            pyopencl_op6_times,
            pyopencl_op7_times):
        w.writerow([a_size, t_op_1, t_op_2,t_op_3, t_op_4,t_op_5, t_op_6,t_op_7 ])




##### PLot against a Variation in D ##### 

python_times=[]
pyopencl_op1_times=[]
pyopencl_op2_times=[]
pyopencl_op3_times=[]
pyopencl_op4_times=[]
pyopencl_op5_times=[]
pyopencl_op6_times=[]
pyopencl_op7_times=[]

H=320
W=480

param=np.arange(1,5,1).astype(np.int32)

for i in param:
        pyopencl_op1_times.append(cl_op1_hist_time(BS,i*D,H,W,4))
        pyopencl_op2_times.append(cl_op2_hist_time(BS,i*D,H,W,4))
        pyopencl_op3_times.append(cl_op3_hist_time(BS,i*D,H,W,4))
        pyopencl_op4_times.append(cl_op4_hist_time(BS,i*D,H,W,4))
        pyopencl_op5_times.append(cl_op5_hist_time(BS,i*D,H,W,4))
        pyopencl_op6_times.append(cl_op6_hist_time(BS,i*D,H,W,4))
        pyopencl_op7_times.append(cl_op7_hist_time(BS,i*D,H,W,4))

plt.clf()
plt.plot(param, pyopencl_op1_times, 'r*-',
         param, pyopencl_op2_times, 'go-',
         param, pyopencl_op3_times, 'bo-',
         param, pyopencl_op7_times, 'mo-'
)

plt.xlabel('Disparity Range')
plt.ylabel('$t$')
plt.title('Execution time as a function of Disparity (Basic optimizations)')
plt.legend(('OP1', 'OP2','OP3', 'OPCV'), loc='upper left')
plt.grid(True)
#plt.gca().set_xlim((min(param*D), max(param*D)))
#plt.gca().set_ylim((0, 1.2*max(pyopencl_op1_times)))
#plt.draw()
plt.savefig('D_scaling_basic.png')

plt.clf()
plt.plot(param, pyopencl_op4_times, 'ro-',
         param, pyopencl_op5_times, 'g*-',
         param, pyopencl_op6_times, 'b*-',
)

plt.xlabel('Disparity Range')
plt.ylabel('$t$')
plt.title('Execution time as a function of Disparity (Custom Memory optimizations)')
plt.legend(('OP4','OP5','OP6'), loc='upper left')
plt.grid(True)
#plt.gca().set_xlim((min(param*D), max(param*D)))
#plt.gca().set_ylim((0, 1.2*max(pyopencl_op1_times)))
#plt.draw()
plt.savefig('D_scaling_Mem_opti.png')

with open('D_scaling.csv', 'w') as f:
    w = csv.writer(f)
    for a_size, t_op_1, t_op_2, t_op_3, t_op_4,t_op_5,t_op_6, t_op_7 in \
        zip(param*param*H*W,
            pyopencl_op1_times,
            pyopencl_op2_times,
            pyopencl_op3_times,
            pyopencl_op4_times,
            pyopencl_op5_times,
            pyopencl_op6_times,
            pyopencl_op7_times):
        w.writerow([a_size, t_op_1, t_op_2,t_op_3, t_op_4,t_op_5, t_op_6,t_op_7 ])

