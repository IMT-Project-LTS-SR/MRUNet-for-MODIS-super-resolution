import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.distance import pdist, squareform
import scipy.optimize as opt
import cv2


def correction_avrg(Temp, TT_unm, iscale):
	
    '''
    This function takes two images (path_temperature and path_unm) and 
    computes the residuals of each coarse and fine pixel. For the coarse pixels 
    it computes the residuals as the difference between the measured coarse 
    temperature of the pixel and the mean of the fine pixel temperatures within 
    the coarse pixel $R_coarse=T_measured - mean(T_unm)$. Then the residuals for each 
    fine pixel resolution inside a coarse pixel is the residual of the coarse 
    pixel (scale invariance hypothesis).
      	
    ############## Inputs:
             
    Temp (large scale): Temperature image. the background pixels must be == 0
        
    TT_unm (fine scale): Sharpened temperature image. the background pixels must be == 0
    
    iscale: size factor between measured pixel and unmixed pixel. 
             Ex: if the initial resolution of Temp is 90m and the resolution 
                 after unmixing is 10m iscale=9. 
            
    ##
    ###
    ######
    ###########
    ############### C. Granero-Belinchon; IMT Atlantique, Brest; 12/2021
    '''

    (rows,cols)=TT_unm.shape	
    
    # We recreate a low resolution temperature image from the unmixed one          
    T_unm_add = cv2.resize(TT_unm, (Temp.shape[1],Temp.shape[0]), interpolation = cv2.INTER_AREA)		
    		
    # Because in the borders of the image we can have background pixels, we have to dont take into account them. 
    # To do that, we take the measured temperatures at coarse scale and impose zero 
    # values to all the pixels of the "Added temperature" that are zero for the measured temperature.		
    mask=np.greater(Temp,0) # We eliminate the background pixels (not in the image)
    T_unm_add[~mask]=0
					
    mask2=np.greater(T_unm_add,0) # We eliminate the background pixels (not in the image)
    Temp[~mask2]=0
	
    # We define the delta at the scale of the measured temperature as in the model.																									
    Delta_T = Temp[0:int(np.floor(rows/iscale)),0:int(np.floor(cols/iscale))] - T_unm_add[0:int(np.floor(rows/iscale)),0:int(np.floor(cols/iscale))]
    
    # We obtain the delta at the scale of the unmixed temperature
    Delta_T_final =np.zeros((rows,cols))
    for ic in range(int(np.floor(cols/iscale))):
        for ir in range(int(np.floor(rows/iscale))):
            for ir_2 in range(ir*iscale,(ir*iscale+iscale)):
                for ic_2 in range(ic*iscale,(ic*iscale+iscale)):
                    Delta_T_final[ir_2,ic_2] = Delta_T[ir,ic]
    
    # We define a mask in order to not use the background pixels. If a pixel of the finer scale image is zero, 
    # the coarse pixel which contains this one is also zero.
    mask_TT_unm=np.zeros((rows,cols))
    mask_TT_unm[np.nonzero(TT_unm)]=1
    
    Delta_T_final=Delta_T_final*mask_TT_unm
		
    # We correct the unmixed temperature using the delta	
    TT_unmixed_corrected = TT_unm + Delta_T_final
    
    print('Correction Done')
	
    return TT_unmixed_corrected

def correction_linreg(index, Temp, TT_unm, iscale, fit_param):
	
    
    '''
    This function takes three images (path_index, path_temperature and path_unm) and 
    computes the residuals of each coarse and fine pixel. For the coarse pixels 
    it computes the residuals as the difference between the measured coarse 
    temperature of the pixel and the coarse temperature obtained with the linear regression 
    $R_coarse=T_measured_coarse - (a1*I_measured_coarse + a0)$. Then the residuals for each 
    fine pixel resolution inside a coarse pixel is the residual of the coarse 
    pixel (scale invariance hypothesis).
      	
    ############## Inputs:
        
    index (large scale): reflective index image, ex: NDVI, NDBI ...
             
    Temp (large scale): Temperature image. the background pixels must be == 0
        
    
    TT_unm (fine scale): Sharpened temperature image. the background pixels must be == 0
    
    iscale: size factor between measured pixel and unmixed pixel. 
             Ex: if the initial resolution of Temp is 90m and the resolution 
                 after unmixing is 10m iscale=9. 
    
    fit_param: fit parameters used to retrieve LST from index
           
    ##
    ###
    ######
    ###########
    ############### C. Granero-Belinchon; IMT Atlantique, Brest; 12/2021
    '''

    (rows,cols)=TT_unm.shape
    
    ## Param[1] indicates that we call the intercept of the linear regression.
    a0 = fit_param[1]
	## Param[0] indicates that we call the slope of the linear regression.
    a1 = fit_param[0]
    
    # We define the matrix of model temperature	
    T_unm_add = a0 + a1*index
            
    mask=np.greater(Temp,0) # We eliminate the background pixels (not in the image)
    T_unm_add[~mask]=0
    
    #  We define the delta at the scale of the measured temperature as in the model.															
    Delta_T = Temp - T_unm_add													
    		
    # We define a mask in order to not use the background pixels. If a pixel of the finer scale image is zero, 
    # the coarse pixel which contains this one is also zero. We have to choose this mask or the mask of line 188
    mask_TT_unm=np.zeros((rows,cols))
    mask_TT_unm[np.nonzero(TT_unm)]=1
    		
    # We obtain the delta at the scale of the unmixed temperature
    Delta_T_final =np.zeros((rows,cols))
    for ic in range(int(np.floor(cols/iscale))):
        for ir in range(int(np.floor(rows/iscale))):
            for ir_2 in range(ir*iscale,(ir*iscale+iscale)):
                for ic_2 in range(ic*iscale,(ic*iscale+iscale)):
                    if mask_TT_unm[ir_2,ic_2]==0:
                        Delta_T_final[ir_2,ic_2] = 0
                    else:
                        Delta_T_final[ir_2,ic_2] = Delta_T[ir,ic]
		
    # We correct the unmixed temperature using the delta	

    TT_unmixed_corrected = TT_unm + Delta_T_final
    
    print('Correction Done')
	
    return TT_unmixed_corrected
	
def quality_correction(Temp, TT_unm, iscale, max_threshold=323, min_threshold=263):
	
    '''
    This function takes two images (path_temperature and path_unm) and 
    computes the residuals of each coarse and fine pixel. For the coarse pixels 
    it computes the residuals as the difference between the measured coarse 
    temperature of the pixel and the mean of the fine pixel temperatures within 
    the coarse pixel $R_coarse=T_measured - mean(T_unm)$. Then the residuals for each 
    fine pixel resolution inside a coarse pixel is the residual of the coarse 
    pixel (scale invariance hypothesis).
    
    Before computing the residuals, this function looks for pixels with temperature values
    above max_thre +5 or below min_thre - 5 and recalculates the temperature of these 
    pixels by averaging the temperatures of the 5x5 squares containing each non-desired 
    temperature pixel in the center.
      	
    ############## Inputs:
        
    index (large scale): reflective index image, ex: NDVI, NDBI ...
             
    Temp (large scale): Temperature image. the background pixels must be == 0
        
    
    TT_unm (fine scale): Sharpened temperature image. the background pixels must be == 0
    
    iscale: size factor between measured pixel and unmixed pixel. 
             Ex: if the initial resolution of Temp is 90m and the resolution 
                 after unmixing is 10m iscale=9. 
         
    ##
    ###
    ######
    ###########
    ############### C. Granero-Belinchon; IMT Atlantique, Brest; 12/2021
    '''
	
    (rows,cols)=TT_unm.shape
    
    ##################### QUALITY CONTROL ##############################
    	
    poids=np.zeros((5,5))
    for i in range(0,5):
        for j in range(0,5):
            poids[i,j]=2*np.sqrt((i-2)**2+(j-2)**2)
    #poids=poids.flatten()

    # Quality control
    ind=np.where(TT_unm > max_threshold+5)
    for i in range(len(ind[0])):
        rowi=ind[0][i]
        coli=ind[1][i]
        
        if rowi>1 and coli>1 and rowi<TT_unm.shape[0]-3 and coli<TT_unm.shape[1]-3 :
            TT_unm[rowi,coli]=np.average(TT_unm[rowi-2:rowi+3,coli-2:coli+3],weights=poids)
    
    ind=np.where((TT_unm < min_threshold-5) & (TT_unm > 0))
    for i in range(len(ind[0])):
        rowi=ind[0][i]
        coli=ind[1][i]
        
        if rowi>1 and coli>1 and rowi<TT_unm.shape[0]-3 and coli<TT_unm.shape[1]-3 :
            TT_unm[rowi,coli]=np.average(TT_unm[rowi-2:rowi+3,coli-2:coli+3],weights=poids)
		
    ################# CORRECTION #######################################

    # We recreate a low resolution temperature image from the unmixed one          
    T_unm_add = cv2.resize(TT_unm, (Temp.shape[1],Temp.shape[0]), interpolation = cv2.INTER_AREA)		
    		
    # Because in the borders of the image we can have background pixels, we have to dont take into account them. 
    # To do that, we take the measured temperatures at coarse scale and impose zero 
    # values to all the pixels of the "Added temperature" that are zero for the measured temperature.		
    mask=np.greater(Temp,0) # We eliminate the background pixels (not in the image)
    T_unm_add[~mask]=0
					
    mask2=np.greater(T_unm_add,0) # We eliminate the background pixels (not in the image)
    Temp[~mask2]=0
	
    # We define the delta at the scale of the measured temperature as in the model.																									
    Delta_T = Temp[0:int(np.floor(rows/iscale)),0:int(np.floor(cols/iscale))] - T_unm_add[0:int(np.floor(rows/iscale)),0:int(np.floor(cols/iscale))]
    
    # We obtain the delta at the scale of the unmixed temperature
    Delta_T_final =np.zeros((rows,cols))
    for ic in range(int(np.floor(cols/iscale))):
        for ir in range(int(np.floor(rows/iscale))):
            for ir_2 in range(ir*iscale,(ir*iscale+iscale)):
                for ic_2 in range(ic*iscale,(ic*iscale+iscale)):
                    Delta_T_final[ir_2,ic_2] = Delta_T[ir,ic]
    
    # We define a mask in order to not use the background pixels. If a pixel of the finer scale image is zero, 
    # the coarse pixel which contains this one is also zero.
    mask_TT_unm=np.zeros((rows,cols))
    mask_TT_unm[np.nonzero(TT_unm)]=1
    
    Delta_T_final=Delta_T_final*mask_TT_unm
		
    # We correct the unmixed temperature using the delta	
    TT_unmixed_corrected = TT_unm + Delta_T_final
    
    print('Correction Done')
	
    return TT_unmixed_corrected
	

def Func_Gamma_cc(pd_uni_c, sill, ran):	
    Gamma_c_temp = sill * (1 - np.exp(-pd_uni_c/(ran/3))) 	# EXPONENTIAL	
    return Gamma_c_temp

def Gamma_ff(pd_uni_c, sill, ran, dis_f, N_c, iscale, pd_c):
    
    Gamma_ff_temp=np.zeros(iscale*iscale)
    Gamma_cc=np.zeros((N_c,N_c))
    	
    for i_coarse in range(N_c):
        for j_coarse in range(N_c):
            temp_var=0
            for i_fine in range(iscale*iscale):
                for i in range(iscale*iscale):
                    Gamma_ff_temp[i] = sill * (1 - np.exp(-dis_f[i_coarse,j_coarse,i_fine,i]/(ran/3))) 	# EXPONENTIAL
                temp_var = sum(Gamma_ff_temp) + temp_var	
            Gamma_cc[i_coarse,j_coarse] = 1/(iscale**4) * temp_var
		
    # We need to classify the Gamma_cc values of pixels i and j in function of the distance between the coarse 
    # i pixel and the coarse j pixel.
		
    Gamma_cc_m=np.zeros(len(pd_uni_c))
    
    for idist in range(len(pd_uni_c)):
        ii=0
        for i_coarse in range(N_c):
            for j_coarse in range(N_c): 	
                if pd_c[i_coarse,j_coarse] == pd_uni_c[idist]:
                    ii=ii+1
                    Gamma_cc_m[idist] = Gamma_cc_m[idist] + Gamma_cc[i_coarse,j_coarse]
            Gamma_cc_m[idist] = Gamma_cc_m[idist]/ii	
		
    Gamma_cc_r=Gamma_cc_m-Gamma_cc_m[0]
		
    return Gamma_cc_r

def correction_ATPRK(index, Temp, TT_unm, fit_param, iscale, scc, block_size, sill, ran, path_plot=False):
	
    '''
    This function takes three images (path_index, path_temperature and path_unm) and 
    computes the residuals of each coarse and fine pixel. For the coarse pixels 
    it computes the residuals as the difference between the measured coarse 
    temperature of the pixel and the coarse temperature obtained with the linear regression 
    $R_coarse=T_measured_coarse - (a1*I_measured_coarse + a0)$. Then the residuals for each 
    fine pixel resolution inside a coarse pixel is computed using the ATPK procedure explained
    in Wang, Shi, Atkinson 2016, 'Area-to-point regression kriging for pan-sharpening'.
      	
    ############## Inputs/Outputs:
    
    index (large scale): reflective index image, ex: NDVI, NDBI ...
             
    Temp (large scale): Temperature image. the background pixels must be == 0
    
    TT_unm (fine scale): Sharpened temperature image. the background pixels must be == 0
    
    fit_param: fit parameters used to retrieve LST from index
    
    iscale: size factor between measured pixel and unmixed pixel. 
            Ex: if the initial resolution of Temp is 90m and the resolution 
                after unmixing is 10m iscale=9. 
    
    scc: The coarse pixel has a size scc * scc
    
    block_size: The moving window used to compute the semivariograms and the weights in the 
                ATPK procedures has a size block_size * block_size 
    
    sill and ran are the parameters defining the shape of the semivariogram. see line 199
    
    path_plot (optional): String with the adress of the fit image of the coarse semivariogram.
    			 The fit must be good for the results to be good.
    
    ###
    ######
    ###########
    ############### C. Granero-Belinchon; IMT Atlantique, Brest; 12/2021
    '''


####################################################################
################ LOADING DATA ######################################
####################################################################

    b_radius = int(np.floor(block_size/2))
	
    (rows,cols)=TT_unm.shape
    (rows_t,cols_t)=Temp.shape
################# CORRECTION #######################################

    ## Param[1] indicates that we call the intercept of the linear regression.
    a0 = fit_param[1]
	## Param[0] indicates that we call the slope of the linear regression.
    a1 = fit_param[0]
    
    # We define the matrix of model temperature	
    T_unm_add = a0 + a1*index
            
    mask=np.greater(Temp,0) # We eliminate the background pixels (not in the image)
    T_unm_add[~mask]=0
    
    #  We define the delta at the scale of the measured temperature as in the model.															
    Delta_T = Temp - T_unm_add			
	
    #####################################################################
    ################ ATPK METHOD ########################################
    #####################################################################
    			
    #####################################################################
    ######### 1 Coarse semivariogram estimation #########################
    #####################################################################
	
	# We define the matrix of distances
    dist=np.zeros((2*block_size-1,2*block_size-1))
    for i in range(-(block_size-1),block_size):
        for j in range(-(block_size-1),block_size):
            dist[i+(block_size-1),j+(block_size-1)]=scc*np.sqrt(i**2+j**2)
	
    # We define the vector of different distances
    distances= np.unique(dist)		
	
    new_matrix=np.zeros(((block_size)**2,3))
	
    Gamma_coarse_temp=np.zeros((rows_t,cols_t,len(distances)))
	
    for irow in range(b_radius,rows_t-b_radius):
        for icol in range(b_radius,cols_t-b_radius):
            dt = Delta_T[irow-b_radius:irow+b_radius+1,icol-b_radius:icol+b_radius+1]	
            for ir in range(block_size):
                for ic in range(block_size):
                    new_matrix[ir*(block_size) + ic,0] = scc*ir
                    new_matrix[ir*(block_size) + ic,1] = scc*ic
                    new_matrix[ir*(block_size) + ic,2] = dt[ir,ic]	
            pd_c = squareform( pdist( new_matrix[:,:2] ) )
            pd_uni_c = np.unique(pd_c)
            N_c = pd_c.shape[0]
            for idist in range(len(pd_uni_c)):
                idd=pd_uni_c[idist]
                if idd==0:
                    Gamma_coarse_temp[irow,icol,idist]=0
                else:
                    ii=0	
                    for i in range(N_c):
                        for j in range(i+1,N_c):
                            if pd_c[i,j] == idd:
                                ii=ii+1
                                Gamma_coarse_temp[irow,icol,idist] = Gamma_coarse_temp[irow,icol,idist] + ( new_matrix[i,2] - new_matrix[j,2] )**2.0
                    Gamma_coarse_temp[irow,icol,idist]	= Gamma_coarse_temp[irow,icol,idist]/(2*ii)		
		
    Gamma_coarse=np.zeros(len(distances))
    for idist in range(len(pd_uni_c)):
        zz=np.nonzero(Gamma_coarse_temp[:,:,idist])
        gg=Gamma_coarse_temp[zz[0],zz[1],idist]
        Gamma_coarse[idist]=np.mean(gg)
	
    Gamma_coarse[np.isnan(Gamma_coarse)]=0	
	
    sillran_c=np.array([sill,ran])
	
    ydata=Gamma_coarse
    xdata=pd_uni_c
	
    param_c, pcov_c = opt.curve_fit(Func_Gamma_cc, xdata, ydata,sillran_c,method='lm')
    
    if path_plot!=False:
        plt.plot(distances,param_c[0] * (1 - np.exp(-distances/(param_c[1]/3))))
        plt.plot(distances,Gamma_coarse)
        plt.savefig(path_plot)
        plt.close()
	
    #####################################################################
    ######### 2 Deconvolution ###########################################
    #####################################################################
    	
    dist_matrix=np.zeros(((iscale*block_size)**2,2))
	
    for ir in range(iscale*block_size):
        for ic in range(iscale*block_size):
            dist_matrix[ir*(iscale*block_size) + ic,0] = scc/iscale * ir
            dist_matrix[ir*(iscale*block_size) + ic,1] = scc/iscale * ic	
	
    pd_f = squareform( pdist( dist_matrix[:,:2] ) )
    #pd_uni_f = np.unique(pd_f)
    #N_f = pd_f.shape[0]
	
    dis_f=np.zeros((N_c,N_c,iscale*iscale,iscale,iscale))
	
    for ii in range(block_size): # We select a coarse pixel
        for jj in range(block_size): # We select a coarse pixel
            #temp_variable=[]
            for iiii in range(iscale): # We take all the fine pixels inside the chosen coarse pixel
                count=0
                for counter in range(jj*iscale+ii*block_size*iscale**2+iiii*block_size*iscale,jj*iscale+ii*block_size*iscale**2+iiii*block_size*iscale+iscale):	
                    res=np.reshape(pd_f[counter,:],(block_size*iscale,block_size*iscale))
                    for icoarse in range(block_size):
                        for jcoarse in range(block_size):
                            dis_f[ii*(block_size)+jj,icoarse*(block_size) + jcoarse,iiii*iscale+count,:,:] = res[icoarse*iscale:icoarse*iscale+iscale,jcoarse*iscale:jcoarse*iscale+iscale]
                    count=count+1		
    
    #Comment: dis_f is a matrix of distances. The first dimension select the coarse i pixel, the second dimension select the coarse j pixel, the third dimension, 
    #         select the fine pixel inside the i coarse pixel and the two last dimensions give us the distance from the fine pixel (third dimension) of i to all 
    #         the fine pixels of j.
	
    ####### We reshape dis_f to convert the last two dimensions iscale, iscale into one dimension iscale*iscale
    dis_f= np.reshape(dis_f,(N_c,N_c,iscale*iscale,iscale*iscale))
		
    sill=param_c[0]
    ran=param_c[1]
    
    sillran=np.array([sill,ran])
	
    ydata=Gamma_coarse
    xdata=pd_uni_c
	
    param, pcov= opt.curve_fit(lambda pd_uni_c, sill, ran: Gamma_ff(pd_uni_c, sill, ran, dis_f, N_c, iscale, pd_c), xdata, ydata, sillran, method='lm') 

	
    #####################################################################
    ######### 3 Estimation Gamma_cc and Gamma_fc ########################
    #####################################################################
	
    ####################### Gamma_cc estimation ######################### 
    Gamma_ff_temp=np.zeros(iscale*iscale)
    Gamma_cc=np.zeros((N_c,N_c))
		
    for i_coarse in range(N_c):
        for j_coarse in range(N_c):
            temp_var=0
            for i_fine in range(iscale*iscale):
                for i in range(iscale*iscale):
                    Gamma_ff_temp[i] = param[0] * (1 - np.exp(-dis_f[i_coarse,j_coarse,i_fine,i]/(param[1]/3))) 	# EXPONENTIAL
                temp_var = sum(Gamma_ff_temp) + temp_var	
            Gamma_cc[i_coarse,j_coarse] =  temp_var/(iscale**4)
    
    # We need to classify the Gamma_cc values of pixels i and j in function of the distance between the coarse 
    # i pixel and the coarse j pixel.
    	
    # Gamma_cc_regularized estimation and comparison with Gamma_coarse	
    Gamma_cc_m=np.zeros(len(pd_uni_c))
    	
    for idist in range(len(pd_uni_c)):
        ii=0
        for i_coarse in range(N_c):
            for j_coarse in range(N_c): 	
                if pd_c[i_coarse,j_coarse] == pd_uni_c[idist]:
                    ii=ii+1
                    Gamma_cc_m[idist] = Gamma_cc_m[idist] + Gamma_cc[i_coarse,j_coarse]
        Gamma_cc_m[idist] = Gamma_cc_m[idist]/ii	
	
    # Gamma_cc_r=Gamma_cc_m-Gamma_cc_m[0]
	
    # gamma_cc_regul compared to gamma_coarse	
    # Diff = Gamma_coarse - Gamma_cc_r
	
    ####################### Gamma_fc estimation #########################
	
    Gamma_ff_temp=np.zeros(iscale*iscale)
    Gamma_fc=np.zeros((iscale*iscale,N_c))
	
    for j_coarse in range(N_c):
        temp_var=0
        for i_fine in range(iscale*iscale):
            for i in range(iscale*iscale):
                Gamma_ff_temp[i] = param[0] * (1 - np.exp(-dis_f[int(np.floor(0.5*block_size**2)),j_coarse,i_fine,i]/(param[1]/3))) 	# EXPONENTIAL
            temp_var = sum(Gamma_ff_temp) 
            Gamma_fc[i_fine,j_coarse] =  temp_var/(iscale**2)
							
    #####################################################################
    ######### 4 Weight estimation (lambdas) #############################
    #####################################################################	
    	
    vec_right = np.ones((block_size**2,1))
    vec_bottom = np.append(np.ones(block_size**2),0)
    	
    A = np.vstack((np.hstack((Gamma_cc,vec_right)),vec_bottom))
    
    Ainv = np.linalg.inv(A)
    	
    B=np.zeros((iscale*iscale,N_c+1))
    lambdas_temp=np.zeros((iscale*iscale,N_c+1))
    for i_fine in range(iscale*iscale):
        B[i_fine,:] = np.append(Gamma_fc[i_fine,:],1) 
        lambdas_temp[i_fine,:] = np.dot(Ainv,B[i_fine,:])	
	
    # lambdas first dimension is the fine pixel inside the central coarse pixel in a (block_size x block_size) pixels block. 
    # lambdas second dimension is the coars pixels in this block.	
    lambdas =lambdas_temp[:,0:len(lambdas_temp[0])-1]
	
    #####################################################################
    ######### 5 Fine Residuals estimation ###############################
    #####################################################################			
	
    mask_TT_unm=np.zeros((rows,cols))
    indice_mask_TT=np.nonzero(TT_unm)
    mask_TT_unm[indice_mask_TT]=1
	
    # We obtain the delta at the scale of the unmixed temperature
    Delta_T_final =np.zeros((rows,cols))
	
    for ic in range(b_radius,int(np.floor(cols/iscale)-b_radius)):
        for ir in range(b_radius,int(np.floor(rows/iscale)-b_radius)):
            for ir_2 in range(ir*iscale,(ir*iscale+iscale)):
                for ic_2 in range(ic*iscale,(ic*iscale+iscale)):
                    if mask_TT_unm[ir_2,ic_2]==0:
                        Delta_T_final[ir_2,ic_2] = 0
                    else:
                        temp_var = np.reshape(Delta_T[ir-b_radius:ir+b_radius+1,ic-b_radius:ic+b_radius+1],block_size**2) # We take the block of coarse pixels
                        # We multiply each coarse pixel in the block by the corresponding lambda and sum
                        Delta_T_final[ir_2,ic_2] = np.sum(lambdas[(ir_2-ir*iscale)*iscale+ic_2-ic*iscale,:]*temp_var)
	
    # We correct the unmixed temperature using the delta	
    TT_unmixed_corrected = TT_unm + Delta_T_final
    
    print('Correction Done')
	
    return TT_unmixed_corrected

def correction_AATPRK(index, Temp, TT_unm, Slope_Intercept, iscale, scc, block_size, sill, ran, path_plot=False):
	
    '''
    This function takes three images (path_index, path_temperature and path_unm) and 
    computes the residuals of each coarse and fine pixel. For the coarse pixels 
    it computes the residuals as the difference between the measured coarse 
    temperature of the pixel and the coarse temperature obtained with the linear regression 
    $R_coarse=T_measured_coarse - (a1*I_measured_coarse + a0)$. Then the residuals for each 
    fine pixel resolution inside a coarse pixel is computed using the ATPK procedure explained
    in Wang, Shi, Atkinson 2016, 'Area-to-point regression kriging for pan-sharpening'.
      	
    ############## Inputs/Outputs:
    
    index (coarse scale): reflective index image, ex: NDVI, NDBI ...
             
    Temp (coarse scale): Temperature image. the background pixels must be == 0
    
    TT_unm (fine scale): Sharpened temperature image. the background pixels must be == 0
    
    Slope_Intercept: Image of dimension [2,:,:] containing Slope and Intercept images. They are used to find LSTcoarse from index
    
    iscale: size factor between measured pixel and unmixed pixel. 
            Ex: if the initial resolution of Temp is 90m and the resolution 
                after unmixing is 10m iscale=9. 
    
    scc: The coarse pixel has a size scc * scc
    
    block_size: The moving window used to compute the semivariograms and the weights in the 
                ATPK procedures has a size block_size * block_size 
    
    sill and ran are the parameters defining the shape of the semivariogram. see line 199
    
    path_plot (optional): String with the adress of the fit image of the coarse semivariogram.
    			 The fit must be good for the results to be good.
    
    ###
    ######
    ###########
    ############### C. Granero-Belinchon; IMT Atlantique, Brest; 12/2021
    '''


    ####################################################################
    ################ LOADING DATA ######################################
    ####################################################################

    b_radius = int(np.floor(block_size/2))
	
    (rows,cols)=TT_unm.shape
    (rows_t,cols_t)=Temp.shape
    ################# CORRECTION #######################################
    
    # We define the matrix of model temperature	
    T_unm_add = Slope_Intercept[1] + Slope_Intercept[0]*index
            
    mask=np.greater(Temp,0) # We eliminate the background pixels (not in the image)
    T_unm_add[~mask]=0
    
    #  We define the delta at the scale of the measured temperature as in the model.															
    Delta_T = Temp - T_unm_add			
	
    #####################################################################
    ################ ATPK METHOD ########################################
    #####################################################################
    			
    #####################################################################
    ######### 1 Coarse semivariogram estimation #########################
    #####################################################################
	
	# We define the matrix of distances
    dist=np.zeros((2*block_size-1,2*block_size-1))
    for i in range(-(block_size-1),block_size):
        for j in range(-(block_size-1),block_size):
            dist[i+(block_size-1),j+(block_size-1)]=scc*np.sqrt(i**2+j**2)
	
    # We define the vector of different distances
    distances= np.unique(dist)		
	
    new_matrix=np.zeros(((block_size)**2,3))
	
    Gamma_coarse_temp=np.zeros((rows_t,cols_t,len(distances)))
	
    for irow in range(b_radius,rows_t-b_radius):
        for icol in range(b_radius,cols_t-b_radius):
            dt = Delta_T[irow-b_radius:irow+b_radius+1,icol-b_radius:icol+b_radius+1]	
            for ir in range(block_size):
                for ic in range(block_size):
                    new_matrix[ir*(block_size) + ic,0] = scc*ir
                    new_matrix[ir*(block_size) + ic,1] = scc*ic
                    new_matrix[ir*(block_size) + ic,2] = dt[ir,ic]	
            pd_c = squareform( pdist( new_matrix[:,:2] ) )
            pd_uni_c = np.unique(pd_c)
            N_c = pd_c.shape[0]
            for idist in range(len(pd_uni_c)):
                idd=pd_uni_c[idist]
                if idd==0:
                    Gamma_coarse_temp[irow,icol,idist]=0
                else:
                    ii=0	
                    for i in range(N_c):
                        for j in range(i+1,N_c):
                            if pd_c[i,j] == idd:
                                ii=ii+1
                                Gamma_coarse_temp[irow,icol,idist] = Gamma_coarse_temp[irow,icol,idist] + ( new_matrix[i,2] - new_matrix[j,2] )**2.0
                    Gamma_coarse_temp[irow,icol,idist]	= Gamma_coarse_temp[irow,icol,idist]/(2*ii)		
		
    Gamma_coarse=np.zeros(len(distances))
    for idist in range(len(pd_uni_c)):
        zz=np.nonzero(Gamma_coarse_temp[:,:,idist])
        gg=Gamma_coarse_temp[zz[0],zz[1],idist]
        Gamma_coarse[idist]=np.mean(gg)
	
    Gamma_coarse[np.isnan(Gamma_coarse)]=0	
	
    sillran_c=np.array([sill,ran])
	
    ydata=Gamma_coarse
    xdata=pd_uni_c
	
    param_c, pcov_c = opt.curve_fit(Func_Gamma_cc, xdata, ydata,sillran_c,method='lm')
    
    if path_plot!=False:
        plt.plot(distances,param_c[0] * (1 - np.exp(-distances/(param_c[1]/3))))
        plt.plot(distances,Gamma_coarse)
        plt.savefig(path_plot)
        plt.close()
	
    #####################################################################
    ######### 2 Deconvolution ###########################################
    #####################################################################
    	
    dist_matrix=np.zeros(((iscale*block_size)**2,2))
	
    for ir in range(iscale*block_size):
        for ic in range(iscale*block_size):
            dist_matrix[ir*(iscale*block_size) + ic,0] = scc/iscale * ir
            dist_matrix[ir*(iscale*block_size) + ic,1] = scc/iscale * ic	
	
    pd_f = squareform( pdist( dist_matrix[:,:2] ) )
    #pd_uni_f = np.unique(pd_f)
    #N_f = pd_f.shape[0]
	
    dis_f=np.zeros((N_c,N_c,iscale*iscale,iscale,iscale))
	
    for ii in range(block_size): # We select a coarse pixel
        for jj in range(block_size): # We select a coarse pixel
            #temp_variable=[]
            for iiii in range(iscale): # We take all the fine pixels inside the chosen coarse pixel
                count=0
                for counter in range(jj*iscale+ii*block_size*iscale**2+iiii*block_size*iscale,jj*iscale+ii*block_size*iscale**2+iiii*block_size*iscale+iscale):	
                    res=np.reshape(pd_f[counter,:],(block_size*iscale,block_size*iscale))
                    for icoarse in range(block_size):
                        for jcoarse in range(block_size):
                            dis_f[ii*(block_size)+jj,icoarse*(block_size) + jcoarse,iiii*iscale+count,:,:] = res[icoarse*iscale:icoarse*iscale+iscale,jcoarse*iscale:jcoarse*iscale+iscale]
                    count=count+1		
    
    #Comment: dis_f is a matrix of distances. The first dimension select the coarse i pixel, the second dimension select the coarse j pixel, the third dimension, 
    #         select the fine pixel inside the i coarse pixel and the two last dimensions give us the distance from the fine pixel (third dimension) of i to all 
    #         the fine pixels of j.
	
    ####### We reshape dis_f to convert the last two dimensions iscale, iscale into one dimension iscale*iscale
    dis_f= np.reshape(dis_f,(N_c,N_c,iscale*iscale,iscale*iscale))
		
    sill=param_c[0]
    ran=param_c[1]
    
    sillran=np.array([sill,ran])
	
    ydata=Gamma_coarse
    xdata=pd_uni_c
	
    param, pcov= opt.curve_fit(lambda pd_uni_c, sill, ran: Gamma_ff(pd_uni_c, sill, ran, dis_f, N_c, iscale, pd_c), xdata, ydata, sillran, method='lm') 

	
    #####################################################################
    ######### 3 Estimation Gamma_cc and Gamma_fc ########################
    #####################################################################
	
    ####################### Gamma_cc estimation ######################### 
    Gamma_ff_temp=np.zeros(iscale*iscale)
    Gamma_cc=np.zeros((N_c,N_c))
		
    for i_coarse in range(N_c):
        for j_coarse in range(N_c):
            temp_var=0
            for i_fine in range(iscale*iscale):
                for i in range(iscale*iscale):
                    Gamma_ff_temp[i] = param[0] * (1 - np.exp(-dis_f[i_coarse,j_coarse,i_fine,i]/(param[1]/3))) 	# EXPONENTIAL
                temp_var = sum(Gamma_ff_temp) + temp_var	
            Gamma_cc[i_coarse,j_coarse] =  temp_var/(iscale**4)
    
    # We need to classify the Gamma_cc values of pixels i and j in function of the distance between the coarse 
    # i pixel and the coarse j pixel.
    	
    # Gamma_cc_regularized estimation and comparison with Gamma_coarse	
    Gamma_cc_m=np.zeros(len(pd_uni_c))
    	
    for idist in range(len(pd_uni_c)):
        ii=0
        for i_coarse in range(N_c):
            for j_coarse in range(N_c): 	
                if pd_c[i_coarse,j_coarse] == pd_uni_c[idist]:
                    ii=ii+1
                    Gamma_cc_m[idist] = Gamma_cc_m[idist] + Gamma_cc[i_coarse,j_coarse]
        Gamma_cc_m[idist] = Gamma_cc_m[idist]/ii	
	
    # Gamma_cc_r=Gamma_cc_m-Gamma_cc_m[0]
	
    # gamma_cc_regul compared to gamma_coarse	
    # Diff = Gamma_coarse - Gamma_cc_r
	
    ####################### Gamma_fc estimation #########################
	
    Gamma_ff_temp=np.zeros(iscale*iscale)
    Gamma_fc=np.zeros((iscale*iscale,N_c))
	
    for j_coarse in range(N_c):
        temp_var=0
        for i_fine in range(iscale*iscale):
            for i in range(iscale*iscale):
                Gamma_ff_temp[i] = param[0] * (1 - np.exp(-dis_f[int(np.floor(0.5*block_size**2)),j_coarse,i_fine,i]/(param[1]/3))) 	# EXPONENTIAL
            temp_var = sum(Gamma_ff_temp) 
            Gamma_fc[i_fine,j_coarse] =  temp_var/(iscale**2)
							
    #####################################################################
    ######### 4 Weight estimation (lambdas) #############################
    #####################################################################	
    	
    vec_right = np.ones((block_size**2,1))
    vec_bottom = np.append(np.ones(block_size**2),0)
    	
    A = np.vstack((np.hstack((Gamma_cc,vec_right)),vec_bottom))
    
    Ainv = np.linalg.inv(A)
    	
    B=np.zeros((iscale*iscale,N_c+1))
    lambdas_temp=np.zeros((iscale*iscale,N_c+1))
    for i_fine in range(iscale*iscale):
        B[i_fine,:] = np.append(Gamma_fc[i_fine,:],1) 
        lambdas_temp[i_fine,:] = np.dot(Ainv,B[i_fine,:])	
	
    # lambdas first dimension is the fine pixel inside the central coarse pixel in a (block_size x block_size) pixels block. 
    # lambdas second dimension is the coars pixels in this block.	
    lambdas =lambdas_temp[:,0:len(lambdas_temp[0])-1]
	
    #####################################################################
    ######### 5 Fine Residuals estimation ###############################
    #####################################################################			
	
    mask_TT_unm=np.zeros((rows,cols))
    indice_mask_TT=np.nonzero(TT_unm)
    mask_TT_unm[indice_mask_TT]=1
	
    # We obtain the delta at the scale of the unmixed temperature
    Delta_T_final =np.zeros((rows,cols))
	
    for ic in range(b_radius,int(np.floor(cols/iscale)-b_radius)):
        for ir in range(b_radius,int(np.floor(rows/iscale)-b_radius)):
            for ir_2 in range(ir*iscale,(ir*iscale+iscale)):
                for ic_2 in range(ic*iscale,(ic*iscale+iscale)):
                    if mask_TT_unm[ir_2,ic_2]==0:
                        Delta_T_final[ir_2,ic_2] = 0
                    else:
                        temp_var = np.reshape(Delta_T[ir-b_radius:ir+b_radius+1,ic-b_radius:ic+b_radius+1],block_size**2) # We take the block of coarse pixels
                        # We multiply each coarse pixel in the block by the corresponding lambda and sum
                        Delta_T_final[ir_2,ic_2] = np.sum(lambdas[(ir_2-ir*iscale)*iscale+ic_2-ic*iscale,:]*temp_var)
	
    # We correct the unmixed temperature using the delta	
    TT_unmixed_corrected = TT_unm + Delta_T_final
    
    print('Correction Done')
	
    return TT_unmixed_corrected
