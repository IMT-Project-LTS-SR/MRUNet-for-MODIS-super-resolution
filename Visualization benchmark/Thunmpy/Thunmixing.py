import numpy as np
import cv2

def linear_unmixing(index, Temp, fit_param, iscale, mask=0):
	
    '''
     This function takes two images (index and temperature) and the linear regression 
     parameters on fit_param and applies $T = a1*I +a0$, where T is the unknown
     a1 and a0 are the regression parameter and I is the index image.
      	
    ############## Inputs:
    
        index (fine scale): reflective index image, ex: NDVI, NDBI ...
          
        Temp (large scale): Temperature image. the background pixels must be == 0
    
        fit_param: Fit parameters from linear model
    
        iscale: Ratio between coarse and fine scale. For coarse =40m and fine =20 iscale =2
    
        mask (fine scale): mask image to put the unmixed temperature values at zero where mask==0. 
                   If mask=0 then the mask is built in this script using the 
                   coarse temperature.
                   
    ############## Outputs:
        
        T_unm= Sharpened LST image 
    ##
    ###
    ######
    ###########
    ############### C. Granero-Belinchon (IMT Atlantique) and A. Michel (Onera); Brest; 12/2021
    '''

    (rows,cols)=index.shape
    
	## Param[1] indicates that we call the intercept of the linear regression.
    a0 = fit_param[1]
	## Param[0] indicates that we call the slope of the linear regression.
    a1 = fit_param[0]
	
    T_unm = a0 + a1*index
	
	### MASK  
    
    if mask == 0:
        maskt=cv2.resize(Temp, (T_unm.shape[1],T_unm.shape[0]), interpolation = cv2.INTER_NEAREST)
        maskt[maskt!=0]=1
        T_unm=T_unm*maskt
    else:
        zz=np.where(mask==0) 
        T_unm[zz]=0
    

    print('Unmixing Done')
	
    return T_unm

	
def bilinear_unmixing(index1, index2, Temp, fit_param, iscale, mask=0):
	
    '''
     This function takes three images (path_index1, path_index2 and path_temperature) and 
     the bilinear regression parameters on the text file (path_fit) and applies 
     $T = a1*I1 +a2*I2 +a0$, where T is the unknown, a2, a1 and a0 are the regression 
     parameters and I1 and I2 are the index images.
      	
    ############## Inputs:
    
        index1 (fine scale): reflective index image, ex: NDVI, NDBI ...
        
        index2 (fine scale): reflective index image, ex: NDVI, NDBI ...
          
        Temp (coarse scale): Temperature image. the background pixels must be == 0
    
        fit_param: Fit parameters from linear model
    
        iscale: Ratio between coarse and fine scale. For coarse =40m and fine =20 iscale =2
    
        mask (fine scale): mask image to put the temperature values at zero where mask==0. 
                   If mask=0 then the mask is built in this script using the 
                   coarse temperature.
                   
    ############## Outputs:
        
        T_unm: Sharpened temperature
    ##
    ###
    ######
    ###########
    ############### C. Granero-Belinchon; IMT Atlantique, Brest; 12/2021
    '''
    
    (rows,cols)=index1.shape
	
    ## Param[0] indicates that we call the intercept of the linear regression.
    p0 = fit_param[0]
    ## Param[1] (Param[2]) indicate that we call the slopes of the linear regression.
    p1 = fit_param[1]
    p2 = fit_param[2]
    		
    T_unm = p0 + p1*index1 + p2*index2
    	
    if mask == 0:
        maskt=cv2.resize(Temp, (T_unm.shape[1],T_unm.shape[0]), interpolation = cv2.INTER_NEAREST)
        maskt[maskt!=0]=1
        T_unm=T_unm*maskt
    else:
        zz=np.where(mask==0) 
        T_unm[zz]=0
    
    print('Unmixing Done')
	
    return T_unm

def linear_unmixing_byclass(index, Temp, Class, fit_param, iscale, mask=0):
	

    '''
     This function takes three images (path_index, path_class and path_temperature) and 
     the linear regression parameters for each class defined in path_class on the 
     text file (path_fit) and applies $T = a1*I +a0$, where T is the unknown, a1 and 
     a0 are the regression parameters (different for each class) and I the index image.
      	
    ############## Inputs:
    
        index (fine scale): reflective index image, ex: NDVI, NDBI ...
          
        Temp (coarse scale): Temperature image. the background pixels must be == 0
        
        Class :  classification image. Background pixels =0
    
        fit_param: Fit parameters from linear model per class
    
        iscale: Ratio between coarse and fine scale. For coarse =40m and fine =20 iscale =2
    
        mask (fine scale): mask image to put the temperature values at zero where mask==0. 
                   If mask=0 then the mask is built in this script using the 
                   coarse temperature.
                   
    ############## Outputs:
        
        T_unm: Sharpened temperature
    
    ###
    ######
    ###########
    ############### C. Granero-Belinchon; IMT Atlantique, Brest; 12/2021
    '''

    (rows,cols)=index.shape

    # We obtain the number of classes and the value characterizing each class
    val_class = np.unique(Class)
    val_class = val_class[np.nonzero(val_class)]
    num_class = len(val_class)
    
    # We initialize the image result
    T_unm=np.zeros((rows,cols))
	
    for i in range(num_class):
        ## Param[1] indicates that we call the intercept of the linear regression.
        a0 = fit_param[1,i]
        ## Param[0] indicates that we call the slope of the linear regression.
        a1 = fit_param[0,i]
	
        for icol in range(cols):
            for irow in range(rows):
                if Class[irow,icol] == val_class[i]:
                    T_unm[irow,icol] = a0 + a1*index[irow,icol]
	
    ### MASK  

    if mask == 0:
        maskt=cv2.resize(Temp, (T_unm.shape[1],T_unm.shape[0]), interpolation = cv2.INTER_NEAREST)
        maskt[maskt!=0]=1
        T_unm=T_unm*maskt
    else:
        zz=np.where(mask==0) 
        T_unm[zz]=0
    
    print('Unmixing Done')
	
    return T_unm

# Multivariate 4th Order Polynomial regression (HUTS)
def huts(x,p1,p2,p3,p4,p5,p6,p7,p8,p9,p10,p11,p12,p13,p14,p15):
	f = p1*x[0,:,:]**4 + p2*x[0,:,:]**3*x[1,:,:] + p3*x[0,:,:]**2*x[1,:,:]**2 + p4*x[0,:,:]*x[1,:,:]**3 + p5*x[1,:,:]**4 +p6*x[0,:,:]**3 + p7*x[0,:,:]**2*x[1,:,:] + p8*x[0,:,:]*x[1,:,:]**2 + p9*x[1,:,:]**3 + p10*x[0,:,:]**2 + p11*x[0,:,:]*x[1,:,:] + p12*x[1,:,:]**2 + p13*x[0,:,:] + p14*x[1,:,:] + p15
	return f

def huts_unmixing(index, albedo, Temp, Param, iscale, mask=0):
	
    '''
    This function takes three images (path_index, path_albedo and path_temperature) and 
    the HUTS regression parameters on the text file (path_fit) and applies 
    the HUTS formula, where T is the unknown, p0, p1 ... p14 are the regression 
    parameters and I and \alpha are the index and albedo images respectively.
      	
    ############## Inputs:
    
        index (fine scale): reflective index image, ex: NDVI, NDBI ...
        
        albedo (fine scale): reflective albedo
          
        Temp (coarse scale): Temperature image. the background pixels must be == 0
        
        Class :  classification image. Background pixels =0
    
        fit_param: Fit parameters from linear model per class
    
        iscale: Ratio between coarse and fine scale. For coarse =40m and fine =20 iscale =2
    
        mask (fine scale): mask image to put the temperature values at zero where mask==0. 
                   If mask=0 then the mask is built in this script using the 
                   coarse temperature.
    
    ############## Outputs:
        
        T_unm: Sharpened temperature
    
    ###
    ######
    ###########
    ############### C. Granero-Belinchon; IMT Atlantique, Brest; 12/2021
    '''
	
    (rows,cols)=index.shape
    
    xdata = np.zeros((2,len(index),len(index[0])))
    xdata[0,:,:] = index
    xdata[1,:,:] = albedo
	
    T_unm = huts(xdata,Param[0],Param[1],Param[2],Param[3],Param[4],Param[5],Param[6],Param[7],Param[8],Param[9],Param[10],Param[11],Param[12],Param[13],Param[14])

	### MASK  

    if mask == 0:
        maskt=cv2.resize(Temp, (T_unm.shape[1],T_unm.shape[0]), interpolation = cv2.INTER_NEAREST)
        maskt[maskt!=0]=1
        T_unm=T_unm*maskt
    else:
        zz=np.where(mask==0) 
        T_unm[zz]=0
    
    print('Unmixing Done')
	
    return T_unm

def aatprk_unmixing(index, Temp, Slope_Intercept, iscale):
	
    '''
    This function takes two images (path_index and path_temperature) and the linear regression 
    parameters on the text file (path_fit) and applies $T = a1*I +a0$, where T is the unknown
    a1 and a0 are the regression parameter and I is the index image.
      	
    ############## Inputs/Outputs:
    
    index (fine scale): reflective index image, ex: NDVI, NDBI ...
          
    Temp (coarse scale): Temperature image. the background pixels must be == 0
    
    Slope_Intercept: Image of dimension [2,:,:] containing Slope and Intercept images. 
    
    iscale: Ratio between coarse and fine scale. For coarse =40m and fine =20 iscale =2
    
    ############## Outputs:
        
    T_unm: Sharpened temperature
    
    ###
    ######
    ###########
    ############### C. Granero-Belinchon; IMT Atlantique, Brest; 12/2021
    '''
    
    (rows,cols)=index.shape
    
    I_mask=np.greater(np.absolute(index),0.0000000000) # We eliminate the background pixels (not in the image)
    T_unm=np.zeros((rows,cols))
    for ic in range(int(np.floor(cols/iscale))):
        for ir in range(int(np.floor(rows/iscale))):
            for ir_2 in range(ir*iscale,(ir*iscale+iscale)):
                for ic_2 in range(ic*iscale,(ic*iscale+iscale)):
                    if I_mask[ir_2,ic_2] == False:		
                        T_unm[ir_2,ic_2]=0			
                    else:
                        T_unm[ir_2,ic_2] = Slope_Intercept[1,ir,ic] + Slope_Intercept[0,ir,ic] * index[ir_2,ic_2]
    

    print('Unmixing Done')
	
    return T_unm