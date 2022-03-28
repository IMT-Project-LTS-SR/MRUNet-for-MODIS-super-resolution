'''
################ Needed packages #### 
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import linregress
import scipy.optimize as opt
'''
import numpy as np
from scipy.stats import linregress
import matplotlib.pyplot as plt
import scipy.optimize as opt

def linear_fit(index, Temp, min_T, path_fit=False, path_plot=False):
    
    '''
    This function takes two images (index and Temp) and computes the linear 
    regression parameters $T = a1*I + a0$. It also applies two masks on the data. 
    The first mask eliminates all the pixels with temperatures below min_T and 
    pixels where Temp==nan, and the second masks eliminates all the pixels 
    with index == nan.
    	
    ############## Inputs:
    
    index: reflective index image, ex: NDVI, NDBI ...
    
    Temp: Temperature image	
    
    min_T: Minimum expected temperature in the image. All the temperatures 
           below this one are considered spurious measures and not taken into account in the 
           regression.	
    
    path_fit: String with the adress where saving the txt file with the regression parameters.
    
    path_plot: String with the adress where saving the plot of T vs I and the linear regression.
    
    ############## Outputs:
    
    fit : Linear fit parameters
    
    #
    ###
    ######
    ###########
    ############### C. Granero-Belinchon; IMT Atlantique, Brest; 12/2021
    '''

########### PROCESSED VERSION OF T AND I AND LINEAR REGRESSION #######
    
    T=Temp.flatten()
    
    T_great_mask=np.greater(T,min_T) # We eliminate the background pixels and those with a temperature under min_T
    T=T[T_great_mask]
    
    I=index.flatten() 
    I=I[T_great_mask]
    	
	# We look for nans in the index and we eliminate those pixels from the index and the temperature
    NaN=np.isfinite(I)
    I=I[NaN]
    T=T[NaN]
    	
# linear regression
    
    fit=linregress(I,T)
    	
###################################################################
########### SAVING RESULTS ########################################
###################################################################
        
    if path_fit != False: 
        np.savetxt(path_fit,(fit[0],fit[1],fit[2],fit[3],fit[4]),fmt='%3.6f')	
        print('Fit parameters saved in file')
    
###################################################################
########### PLOT ##################################################
###################################################################
    
    if path_plot != False: 
        		
        plt.figure(1)
        plt.scatter(I,T,s=1,c='blue')
        plt.plot(I,fit[0]*I+fit[1],'--k')
        plt.xlabel('I')
        plt.ylabel('T')
        plt.grid()
        plt.savefig(path_plot)
        plt.close() 
        
        print('Fit Done')
        		
    else:
        		
        print('Fit Done')
	
    return fit


################# DEFINITION OF THE BILINEAR EXPRESSION ##############
	
def bilinear(x,p0,p1,p2):
	f = p1*x[:,0] + p2*x[:,1] + p0
	return f

def bilinear_fit(index1, index2, Temp, min_T, path_fit):
	
    '''
    This function takes three images (path_index1, apth_index2 and path_temperature) and 
    computes the linear regression parameters $T = a1*I1 +a2*I2 + a0$. It also 
    applies two masks on the data. The first mask eliminates all the 
    pixels with temperatures below min_T and pixels where Temp==nan, 
    and the second masks eliminates all the pixels with index1 == nan or index2 ==nan.
      	
    ############## Inputs:
    
    index: reflective index image, ex: NDVI, NDBI ...
    
    index2: reflective index image, ex: NDVI, NDBI ...
    
    Temp: Temperature image	
    
    min_T: Minimum expected temperature in the image. All the temperatures 
           below this one are considered spurious measures and not taken into account in the 
           regression.	
    
    path_fit: String with the adress where saving the txt file with the regression parameters.
    
    ############## Outputs:
    
    param: bilinear fit parameters
        
    ###
    ######
    ###########
    ###############  C. Granero-Belinchon; IMT Atlantique, Brest; 12/2021
    '''

########### PROCESSED VERSION OF T AND I AND LINEAR REGRESSION #######

    T=Temp.flatten()

    T_great_mask=np.greater(T,min_T) # We eliminate the background pixels and those with a temperature under min_T
    T=T[T_great_mask]

    I1=index1.flatten() 
    I1=I1[T_great_mask]
    I2=index2.flatten() 
    I2=I2[T_great_mask]
    
    # We look for nans in the index and we eliminate those pixels from the index and the temperature
    NaN=np.isfinite(I1)
    I1=I1[NaN]
    I2=I2[NaN]
    T=T[NaN]
    
    NaN=np.isfinite(I2)
    I1=I1[NaN]
    I2=I2[NaN]
    T=T[NaN]
	
################ LINEAR REGRESSION ################################## 

    ydata = T
    xdata = np.zeros((len(I1),2))
    xdata[:,0] = I1
    xdata[:,1] = I2

    p = [1.,1.,1.]

    param, pcov = opt.curve_fit(bilinear, xdata, ydata,p,method='lm')
	
###################################################################
########### SAVING RESULTS ########################################
###################################################################

    if path_fit != False:
        np.savetxt(path_fit,(param),fmt='%3.6f')		
        print('Fit parameters saved in file')
    
###################################################################
########### PLOT ##################################################
###################################################################
		
    print('Fit Done')
	
    return param
	
def linear_fit_window(index, Temp, min_T, b_radius):
	
    '''
    This function takes two images (index and temperature) and 
    computes the linear regression parameters $T = a1*I + a0$ within a moving 
    window of size (2*b_radius+1)^2. The obtained slope and intercept values 
    are assigned to the center pixel of the window. It also 
    applies two masks on the data. The first mask eliminates all the 
    pixels with temperatures below min_T and pixels where Temp==nan, 
    and the second masks eliminates all the pixels with index == nan. 
    The code returns two images, a0 and a1 images, equal in size to the input images.
     	
    ############## Inputs:
    
    index: reflective index image, ex: NDVI, NDBI ...
    
    Temp: Temperature image
    
    min_T: Minimum expected temperature in the image. All the temperatures 
           below this one are considered spurious measures and not taken into account in the 
           regression.	
    
    b_radius: The moving window used to compute the regression parameters
                has a size (2*b_radius+1)^2 
    
    ############## Outputs:

    a0: Image with intercepts
    a1: Image with slopes
        
    #
    ##
    ###
    ######
    ###########
    ############### C. Granero-Belinchon (IMT Atlantique) and A. Michel (Onera); Brest; 12/2021
    '''

########### PROCESSED VERSION OF T AND I AND LINEAR REGRESSION #######
    
    (rows,cols)=Temp.shape

    T=Temp.flatten()

    T_great_mask=np.greater(T,min_T) # We eliminate the background pixels and those with a temperature under min_T
    T=T[T_great_mask]

    I=index.flatten() 
    I=I[T_great_mask]
	
    # We look for nans in the index and we eliminate those pixels from the index and the temperature
    NaN=np.isfinite(I)
    I=I[NaN]
    T=T[NaN]
	
# Two different linear regressions

    fit2=linregress(I,T)
	
#####################################################################
################### WINDOW DEFINITION , #############################
########### AND LINEAR REGRESSION FOR EACH WINDOW ###################

    a1 = np.zeros((rows, cols))
    a0 = np.zeros((rows, cols))

    for irow in range(b_radius,rows-b_radius):
        print(irow, end=" ")
        for icol in range(b_radius,cols-b_radius):
            
            Tw = Temp[irow-b_radius:irow+b_radius+1,icol-b_radius:icol+b_radius+1]
            Iw = index[irow-b_radius:irow+b_radius+1,icol-b_radius:icol+b_radius+1]
            Tw = Tw.flatten()
            Iw = Iw.flatten()
            mask=np.greater(Tw,min_T) # We eliminate the background pixels and those with a temperature under min_T K
            Tw=Tw[mask]
            Iw=Iw[mask]
            NaN=np.isfinite(Iw) # We look for nans in the index and we eliminate those pixels from the index and the temperature
            Iw=Iw[NaN]
            Tw=Tw[NaN]
            if len(Tw) > 2/3*(2*b_radius+1)**2:
                fit=linregress(Iw,Tw)
                a1[irow,icol]=fit[0]
                a0[irow,icol]=fit[1]
            if len(Tw) <= 2/3*(2*b_radius+1)**2:
                a1[irow,icol]=fit2[0]
                a0[irow,icol]=fit2[1]
	
    for irow in range(0,b_radius):
        for icol in range(cols):			
            a1[irow,icol]=fit2[0]
            a0[irow,icol]=fit2[1]
    
    for irow in range(rows-b_radius,rows):
        for icol in range(cols):			
            a1[irow,icol]=fit2[0]
            a0[irow,icol]=fit2[1]
    			
    for icol in range(0,b_radius):
        for irow in range(rows):			
            a1[irow,icol]=fit2[0]
            a0[irow,icol]=fit2[1]
			
    for icol in range(cols-b_radius,cols):
        for irow in range(rows):			
            a1[irow,icol]=fit2[0]
            a0[irow,icol]=fit2[1]
    

    print('Fit Done')
	
    return a0, a1

# Multivariate 4th Order Polynomial regression (HUTS)
def huts(x,p1,p2,p3,p4,p5,p6,p7,p8,p9,p10,p11,p12,p13,p14,p15):
	f = p1*x[:,0]**4 + p2*x[:,0]**3*x[:,1] + p3*x[:,0]**2*x[:,1]**2 + p4*x[:,0]*x[:,1]**3 + p5*x[:,1]**4 +p6*x[:,0]**3 + p7*x[:,0]**2*x[:,1] + p8*x[:,0]*x[:,1]**2 + p9*x[:,1]**3 + p10*x[:,0]**2 + p11*x[:,0]*x[:,1] + p12*x[:,1]**2 + p13*x[:,0] + p14*x[:,1] + p15
	return f

def huts_fit(index, albedo, Temp, min_T, p=False, path_fit=False):
	
    '''
    This function takes three images (index, albedo and temperature) and 
    computes the polynomial regression parameters of HUTS model. It also 
    applies two masks on the data. The first mask eliminates all the 
    pixels with temperatures below min_T and where T==nan, and the second masks 
    eliminates all the pixels with index1 == nan.
      	
    ############## Inputs:
    
    index: reflective index image, ex: NDVI, NDBI ...
    
    albedo: albedo from reflective domain
    
    Temp: Temperature image	
    
    min_T: Minimum expected temperature in the image. All the temperatures 
           below this one are considered spurious measures and not taken into account in the 
           regression.	
           
    p : initial guess for fit parameters
    
    path_fit: String with the adress where saving the txt file with the regression parameters.
    
    ############## Outputs:
    
    ##
    ###
    ######
    ###########
    ############### C. Granero-Belinchon; IMT Atlantique, Brest; 12/2021
    '''

########### PROCESSED VERSION OF T AND I AND LINEAR REGRESSION #######

    T=Temp.flatten()

    T_great_mask=np.greater(T,min_T) # We eliminate the background pixels and those with a temperature under min_T
    T=T[T_great_mask]

    I=index.flatten() 
    I=I[T_great_mask]
    A=albedo.flatten() 
    A=A[T_great_mask]
    
    # We look for nans in the index and we eliminate those pixels from the index and the temperature
    NaN=np.isfinite(I)
    I=I[NaN]
    A=A[NaN]
    T=T[NaN]
	
    NaN=np.isfinite(A)
    I=I[NaN]
    A=A[NaN]
    T=T[NaN]
	
################ POLYNOMIAL REGRESSION ################################## 

    ydata = T 
    xdata = np.zeros((len(I),2))
    xdata[:,0] = I
    xdata[:,1] = A

    if p==False:
        p = [1.,1.,1.,1.,1.,1.,1.,1.,1.,1.,1.,1.,1.,1.,1.]

    param, pcov = opt.curve_fit(huts, xdata, ydata,p,method='lm')

###################################################################
########### SAVING RESULTS ########################################
###################################################################

    if path_fit != False:        
        np.savetxt(path_fit,(param),fmt='%3.6f')		
        print('Fit parameters saved in file')

    print('Fit Done')
	
    return param

def fit_byclass(index, Temp, Class, min_T, path_fit=False, path_plot=False):
	
    '''
    This function takes three images (index, class and temperature) and 
    computes the linear regression parameters $T = a1*I +a0$ separately for 
    each class defined in Class. It also applies two masks on the data. 
    The first mask eliminates all the pixels with temperatures below min_T 
    and where T==nan, and the second masks eliminates all the pixels with 
    index == nan.
      	
    ############## Inputs:
    
    index: reflective index image, ex: NDVI, NDBI ...
    
    Temp: Temperature image	
    
    Class : Classification image file. Background pixels =0
    
    min_T: Minimum expected temperature in the image. All the temperatures 
           below this one are considered spurious measures and not taken into account in the 
           regression.	
    
    path_fit: String with the adress where saving the txt file with the regression parameters.
    
    path_plot: String with the adress where saving the plot of T vs I and the linear regression.
    
    ############## Outputs:
    
    fit_c: matrix of size (number of classes x fit parameters) with the fit parameters per class
        
    ##
    ###
    ######
    ###########
    ############### C. Granero-Belinchon; IMT Atlantique, Brest and A. Michel; DOTA-ONERA, Toulouse; 12/2021
    '''

# We obtain the number of classes and the value characterizing each class
    val_class = np.unique(Class)
    val_class = val_class[np.nonzero(val_class)]
    num_class = len(val_class)

########### FINAL VERSION OF T AND I AND LINEAR REGRESSION ##########

    T = Temp.flatten()
    I = index.flatten() 
    C = Class.flatten()
	
# Classification 

    fit_c=np.zeros((num_class,5))
    #creation of the header for the filetext
    linehead = ''
    
    for i in range(num_class):
        linehead = linehead + 'Class'+str(i+1)+' ' #creation of the header for the filetext
        cc = np.where(C==val_class[i])
        Ic = I[cc]
        Tc = T[cc]
		
	# Mask 
			
        T_great_mask_c=np.greater(Tc,min_T) # We eliminate the background pixels and those with a temperature under 285K
        Tc=Tc[T_great_mask_c]
        Ic=Ic[T_great_mask_c]
        NaN=np.isfinite(Ic)
        Ic=Ic[NaN]
        Tc=Tc[NaN]
		
        fit_c[i]=linregress(Ic,Tc)

###################################################################
########### PLOT ##################################################
###################################################################

        if path_plot !=False:

            plt.figure(1)
            plt.scatter(Ic,Tc,s=1,c='blue')
            plt.plot(Ic,fit_c[i,0]*Ic + fit_c[i,1],'--k')
            plt.xlabel('I')
            plt.ylabel('T')
            plt.grid()
            plt.savefig(path_plot[:-4] + "_class"+ str(i+1) + path_plot[-4:])
            plt.close()
			
            print('Fit by class Done')
        else:
            print('Fit by class Done')
	
    if path_fit!=False:
        np.savetxt(path_fit,(fit_c[:,0],fit_c[:,1],fit_c[:,2],fit_c[:,3],fit_c[:,4]),header=linehead,fmt='%3.6f')
        
    return fit_c.T
