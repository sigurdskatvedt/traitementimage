# Similarity Measures 

import tifffile
import numpy as np
import sys
import time

class Similarity:
    def __init__(self,im1,im2,disparityRange):
        self.imLName=im1
        self.imRName=im2
        self.dispRange=disparityRange
        
        
########################################################################

        
    def ComputeSim_SSD(self,WIN_SIZE,disparityNameFile):
        #load both images
        Im1=tifffile.imread(self.imLName)
        Im2=tifffile.imread(self.imRName)
        #compute similarity Measure for all the specified disparity range
        Disparity=np.zeros(Im1.shape,dtype='float')
        
        print(Im1.shape[0])
        
        """
        TO DO  !
        
        COMPUTE THE BEST DISPARITY VALUE i.e : THE DISPARITY THAT MINIMZES THE SSD CORRELATION CRITERION
        
        Tips:
        
        Loop on all image pixels (take into account the correlation window size ) : borders !
            To search for the best disparity for each pixel you need to compute the correlation criterion for all the disparity candidates, so:
                loop on all disparity range (the search space)
                    compute the correlation criterion for each disparity and anc compare to the minimum (for SSD)
                
                Get the best disparity that minimizes SSD 
                STORE IT ! 
        
        """
        win_movement = (WIN_SIZE-1) / 2
        for x in range(Im2.shape[0]):
            for y in range(Im2.shape[1]):
                for d in self.dispRange:
                    SSD_sum = 0
                    for i in range(x-win_movement, x+win_movement):
                        for j in range(y-win_movement,x+win_movement):
                            SSD_sum += (Im1[x+j,y+i] - Im2[x+j,y+i]) ** 2


        
        # Store the best disparity image based only on similarity values 
        # tifffile.imsave(disparityNameFile,Disparity)
        
        
########################################################################
    def ComputeSim_NCC(self,WIN_SIZE,disparityNameFile):   
        #load both images
        Im1=tifffile.imread(self.imLName)
        Im2=tifffile.imread(self.imRName)
        #compute similarity Measure for all the specified disparity range
        Disparity=np.zeros(Im1.shape,dtype='float')
        
        
        
        """
        TO DO ! 
        
        Implement NCC correlator between a pair of images and store the resulting disparity map
        
        IDEA: Normalize patches with mean and standard deviation and compute the NCC measure
        
        
        """  
        # Store the best disparity image based only on similarity values 
        tifffile.imsave(disparityNameFile,Disparity)
          


if __name__=="__main__":
    image1Name="./IMAGES/ImL_Scaled_Scaled.tif"
    image2Name="./IMAGES/ImR_Scaled_Scaled.tif"
    DispName="./IMAGES/Disparity.tif"
	
    CompSimilarity=Similarity(image1Name,image2Name,120)
    CompSimilarity.ComputeSim_SSD(None, DispName)
