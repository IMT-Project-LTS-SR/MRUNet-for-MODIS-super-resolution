import sys
import cv2
sys.path.insert(0,".") 
from Thunmpy import ThunmFit
import numpy as np
import matplotlib.pyplot as plt
from skimage.metrics import structural_similarity as SSIM
from osgeo import gdal, gdalconst
import math
from scipy.spatial.distance import pdist, squareform
import scipy.optimize as opt
import torch

def normalize(image, mean=290.6732, std=12.0763):
  return (image - mean)/std
  
def denormalize(image, mean=290.6732, std=12.0763):
  return image*std + mean

def downsampling(img, scale):
  down_img = np.zeros((int(img.shape[0]/scale), int(img.shape[1]/scale)))
  for y in range(0, img.shape[0], scale):
    for x in range(0, img.shape[1], scale):
      window = img[y:y+scale, x:x+scale].reshape(-1)
      # sum4 = np.sum(window**4)
      # if sum4 < 0.000001:
      #   norm4 = 0
      # else:
      #   norm4 = sum4/len(window[window != 0.0])
      # norm4 = norm4**0.25
      if 0 not in window:
        sum4 = np.sum(window**4)
        norm4 = (sum4/(scale**2))**0.25
      else:
        norm4 = 0.0
      down_img[int(y/scale), int(x/scale)] = norm4
  
  return down_img
  
def ssim(label, outputs):
    # label = label.reshape(label.shape[0], label.shape[2], label.shape[3])
    # outputs = outputs.reshape(outputs.shape[0], outputs.shape[2], outputs.shape[3])
    # outputs *= max_val

    ssim_map = SSIM(label, outputs, data_range=label.max() - label.min())
    # C1 = (0.01 * np.max(labe))**2
    # C2 = (0.03 * np.max(labe))**2

    # img1 = labe.astype(np.float64)
    # img2 = out.astype(np.float64)
    # kernel = cv2.getGaussianKernel(7, 1.5)
    # window = np.outer(kernel, kernel.transpose())

    # mu1 = cv2.filter2D(img1, -1, window)[5:-5, 5:-5]  # valid
    # mu2 = cv2.filter2D(img2, -1, window)[5:-5, 5:-5]
    # mu1_sq = mu1**2
    # mu2_sq = mu2**2
    # mu1_mu2 = mu1 * mu2
    # sigma1_sq = cv2.filter2D(img1**2, -1, window)[5:-5, 5:-5] - mu1_sq
    # sigma2_sq = cv2.filter2D(img2**2, -1, window)[5:-5, 5:-5] - mu2_sq
    # sigma12 = cv2.filter2D(img1 * img2, -1, window)[5:-5, 5:-5] - mu1_mu2

    # ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) *
    #                                                       (sigma1_sq + sigma2_sq + C2))
    # ssim_maps.append(ssim_map)
    return ssim_map

def psnr_notorch(label, outputs):
    """
    Compute Peak Signal to Noise Ratio (the higher the better).
    PSNR = 20 * log10(MAXp) - 10 * log10(MSE).
    https://en.wikipedia.org/wiki/Peak_signal-to-noise_ratio#Definition
    First we need to convert torch tensors to NumPy operable.
    """
    # label = label.cpu().detach().numpy()
    # outputs = outputs.cpu().detach().numpy()
    img_diff = outputs - label
    rmse = math.sqrt(np.mean((img_diff) ** 2))
    if rmse == 0 or rmse==np.inf:
        return 0, np.inf
    else:
        PSNR = 20 * np.log10((np.max(label) - np.min(label)) / rmse)
        return PSNR, rmse
        
def read_tif(in_file):
  """
  Similar to read_modis but this function is for reading output .tif file from
  """
  # save_tif()

  dataset =  gdal.Open(in_file, gdal.GA_ReadOnly)

  cols =dataset.RasterXSize
  rows = dataset.RasterYSize
  projection = dataset.GetProjection()
  geotransform = dataset.GetGeoTransform()

  # Coordinates of top left pixel of the image (Lat, Lon)
  coords=np.asarray((geotransform[0],geotransform[3]))

  # Open dataset Day as np.array()
  band = dataset.GetRasterBand(1)
  LST_K_day = band.ReadAsArray().astype(np.float)
  bandtype = gdal.GetDataTypeName(band.DataType)

  # open dataset Night as np.array()
  band = dataset.GetRasterBand(2)
  LST_K_night = band.ReadAsArray().astype(np.float)
  bandtype = gdal.GetDataTypeName(band.DataType)

  return LST_K_day,LST_K_night,cols,rows,projection,geotransform

def read_tif_MOD13A2(in_file):
  """
  Similar to read_modis but this function is for reading output .tif file from
  """
  # save_tif()

  dataset =  gdal.Open(in_file, gdal.GA_ReadOnly)

  cols =dataset.RasterXSize
  rows = dataset.RasterYSize
  projection = dataset.GetProjection()
  geotransform = dataset.GetGeoTransform()

  # Coordinates of top left pixel of the image (Lat, Lon)
  coords=np.asarray((geotransform[0],geotransform[3]))

  # Open dataset red as np.array()
  band = dataset.GetRasterBand(1)
  red = band.ReadAsArray().astype(np.float)
  bandtype = gdal.GetDataTypeName(band.DataType)

  # open dataset NIR as np.array()
  band = dataset.GetRasterBand(2)
  NIR = band.ReadAsArray().astype(np.float)
  bandtype = gdal.GetDataTypeName(band.DataType)

  # open dataset NIR as np.array()
  band = dataset.GetRasterBand(3)
  MIR = band.ReadAsArray().astype(np.float)
  bandtype = gdal.GetDataTypeName(band.DataType)

  return red, NIR, MIR,cols,rows,projection,geotransform


def linear_fit_test(path_index, path_temperature, min_T, path_fit, plot, path_plot):
    from osgeo import gdal
    import numpy as np
    from scipy.stats import linregress
    import matplotlib.pyplot as plt
################### INDEX ##########################################
    
# Reading index data

    cols = path_index.shape[0] #Spatial dimension x
    rows = path_index.shape[1] #Spatial dimension y

# Read as array the index
    index = path_index

################### Temperature ######################################

# Reading temperature data

    cols_t = path_temperature.shape[0] #Spatial dimension x
    rows_t = path_temperature.shape[1] #Spatial dimension y

# Read as array
    Temp= path_temperature


########### PROCESSED VERSION OF T AND I AND LINEAR REGRESSION #######

    T=Temp.flatten()

    T_great_mask=np.greater(T,0) # We eliminate the background pixels and those with a temperature under min_T
    T=T[T_great_mask]

    I=index.flatten() 
    I=I[T_great_mask]
    
    # We look for nans in the index and we eliminate those pixels from the index and the temperature
    # NaN=np.isfinite(I)
    # I=I[NaN]
    # T=T[NaN]
    
# Two different linear regressions

    fit = np.polyfit(I,T,1)
    fit_fn = np.poly1d(fit) 
    fit2=linregress(I,T)
    
###################################################################
########### SAVING RESULTS ########################################
###################################################################
                
    np.savetxt(path_fit,(fit2[0],fit2[1],fit2[2],fit2[3],fit2[4]),fmt='%3.6f')    

###################################################################
########### PLOT ##################################################
###################################################################

    if plot == 1: 
        
        # plt.figure(1)
        # plt.scatter(I,T,s=1,c='blue')
        # plt.plot(I,fit_fn(I),'--k')
        # plt.xlabel('I')
        # plt.ylabel('T')
        # plt.grid()
        # plt.savefig(path_plot)
        # plt.close()
        
        print('Fit Done')
        
    else:
        pass
    #     print('Fit Done')
    # print("index",index.shape)
    # print("Temp",Temp.shape)
    return


def linear_unmixing_test(path_index, path_temperature, path_fit, iscale, path_mask, path_out):
    
    import numpy as np
    from osgeo import gdal, gdalconst
    
####################################################################
################ LOADING DATA ######################################
####################################################################

################### INDEX ########################################## # fine scale
    cols = path_index.shape[1] #Spatial dimension x
    rows = path_index.shape[0] #Spatial dimension y
# Read as array the index
    index = path_index

################### Temperature ######################################

# Reading temperature data

    cols_t = path_temperature.shape[1] #Spatial dimension x
    rows_t = path_temperature.shape[0] #Spatial dimension y

# Read as array
    Temp= path_temperature

################### Mask ###################################### # fine scale

    if path_mask !=0:
# Reading mask data
        MaskFile = path_mask
        dataset_Mask = gdal.Open(MaskFile)

        cols_m = dataset_Mask.RasterXSize #Spatial dimension x
        rows_m = dataset_Mask.RasterYSize #Spatial dimension y

# Read as array
        mask= dataset_Mask.ReadAsArray(0, 0, cols_m, rows_m).astype(np.float)

###################################################################
########### UNMIXING ##############################################
###################################################################
    
    Param = np.genfromtxt(path_fit)
    
    ## Param[1] indicates that we call the intercept of the linear regression.
    a0 = Param[1]
    ## Param[0] indicates that we call the slope of the linear regression.
    a1 = Param[0]
    
    T_unm = a0 + a1*index
    
    ### MASK  

    if path_mask == 0:

        for ir in range(int(np.floor(rows/iscale))):
            for ic in range(int(np.floor(cols/iscale))):
                if Temp[ir,ic] == 0:
                    for ir_2 in range(ir*iscale,(ir*iscale+iscale)):
                        for ic_2 in range(ic*iscale,(ic*iscale+iscale)):
                            T_unm[ir_2,ic_2]=0
    else:
        zz=np.where(mask==0) 
        T_unm[zz]=0
                
    # This mask doesn't work because we don't have acces to the 20m temperature                    
    #I_mask = np.greater(Temp_jour_20m,0) # We eliminate the background pixels (not in the image)
    #for icol in range(cols):
    #    for irow in range(rows):
    #        if I_mask[irow,icol] == False:
    #            T_unm[irow,icol]=0
        
###################################################################
########### SAVING RESULTS ########################################
###################################################################
    
    # print('Unmixing Done')
    # print("index",index.shape)
    # print("Temp",Temp.shape)
    return T_unm


import numpy as np
import matplotlib.pyplot as plt
from osgeo import gdal, gdalconst
import math
from scipy.spatial.distance import pdist, squareform
import scipy.optimize as opt    

def correction_ATPRK_test(path_index, path_fit, path_temperature, path_unm, iscale, scc, block_size, sill, ran, path_out, path_out1, path_out2, path_out3,path_plot,T_unm):
    

####################################################################
################ LOADING DATA ######################################
####################################################################

    b_radius = int(np.floor(block_size/2))
    
################### INDEX Coarse ###################################

    cols_i = path_index.shape[1] #Spatial dimension x
    rows_i = path_index.shape[0] #Spatial dimension y
# Read as array the index
    index = path_index

    cols_t = path_temperature.shape[1] #Spatial dimension x
    rows_t = path_temperature.shape[0] #Spatial dimension y
    Temp= path_temperature


    Tm=Temp
#     TempFile = '/content/drive/MyDrive/Ganglin/1_IMT/MCE_Projet3A/MODIS/MOD_2020/tifs_files/2km_of_MOD11A1.A2020001.h18v04.061.2021003092415.hdf.0015.tif'
#     dataset_Temp = gdal.Open(TempFile)

#     dtset=dataset_Temp
    
# ################### UNMIXED Temperature ###########################        

# #dataset
#     T_unmix = gdal.Open(path_unm)

    cols = T_unm.shape[1] #Spatial dimension x
    rows = T_unm.shape[0] #Spatial dimension y

# Read as array
    TT_unm = T_unm
        
################# CORRECTION #######################################

# We define the temperature at the coarse scale obtained from the regression.
    Param = np.genfromtxt(path_fit)
# We take the parameters at the coarse scale
    a0 = Param[1]
    a1 = Param[0]
        # We define the matrix of added temperature    
    T_unm_add =np.zeros((rows_t,cols_t))
    mask=np.greater(Tm,0) # We eliminate the background pixels (not in the image)
        
    for i in range(rows_t):
        for j in range(cols_t):
            T_unm_add[i,j] = a0 + a1*index[i,j]
            if mask[i,j] == False:
                T_unm_add[i,j] = 0
        
# We define the delta at the scale of the measured temperature as in the model.        
                                                                                                        
    Delta_T = Tm - T_unm_add            
    
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
            dist[i+(block_size-1),j+(block_size-1)]=20*iscale*np.sqrt(i**2+j**2)
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
                    Gamma_coarse_temp[irow,icol,idist]    = Gamma_coarse_temp[irow,icol,idist]/(2*ii)        
    
    #Gamma_coarse_temp[np.isnan(Gamma_coarse_temp)]=0    
    
    Gamma_coarse=np.zeros(len(distances))
    for idist in range(len(pd_uni_c)):
        zz=np.nonzero(Gamma_coarse_temp[:,:,idist])
        gg=Gamma_coarse_temp[zz[0],zz[1],idist]
        Gamma_coarse[idist]=np.mean(gg)
    
    Gamma_coarse[np.isnan(Gamma_coarse)]=0    
    
    def Func_Gamma_cc(pd_uni_c, sill, ran):
        
        Gamma_c_temp = sill * (1 - np.exp(-pd_uni_c/(ran/3)))     # EXPONENTIAL
        
        return Gamma_c_temp
    
    sillran_c=np.array([sill,ran])
    
    ydata=Gamma_coarse
    xdata=pd_uni_c
    
    param_c, pcov_c = opt.curve_fit(Func_Gamma_cc, xdata, ydata,sillran_c,method='lm')
    # plt.plot(distances,param_c[0] * (1 - np.exp(-distances/(param_c[1]/3))))
    # plt.plot(distances,Gamma_coarse)
    # plt.savefig(path_plot)
    # plt.close()
    
#####################################################################
######### 2 Deconvolution ###########################################
#####################################################################
    
    dist_matrix=np.zeros(((iscale*block_size)**2,2))
    
    for ir in range(iscale*block_size):
        for ic in range(iscale*block_size):
            dist_matrix[ir*(iscale*block_size) + ic,0] = scc/iscale * ir
            dist_matrix[ir*(iscale*block_size) + ic,1] = scc/iscale * ic    
    
    pd_f = squareform( pdist( dist_matrix[:,:2] ) )
    pd_uni_f = np.unique(pd_f)
    N_f = pd_f.shape[0]
    
    dis_f=np.zeros((N_c,N_c,iscale*iscale,iscale,iscale))
    
    for ii in range(block_size): # We select a coarse pixel
        for jj in range(block_size): # We select a coarse pixel
            temp_variable=[]
            for iiii in range(iscale): # We take all the fine pixels inside the chosen coarse pixel
                count=0
                for counter in range(jj*iscale+ii*block_size*iscale**2+iiii*block_size*iscale,jj*iscale+ii*block_size*iscale**2+iiii*block_size*iscale+iscale):    
                    res=np.reshape(pd_f[counter,:],(block_size*iscale,block_size*iscale))
                    for icoarse in range(block_size):
                        for jcoarse in range(block_size):
                            dis_f[ii*(block_size)+jj,icoarse*(block_size) + jcoarse,iiii*iscale+count,:,:] = res[icoarse*iscale:icoarse*iscale+iscale,jcoarse*iscale:jcoarse*iscale+iscale]
                    count=count+1        
    
####### Comment: dis_f is a matrix of distances. The first dimension select the coarse i pixel, the second dimension select the coarse j pixel, the third dimension, 
#######          select the fine pixel inside the i coarse pixel and the two last dimensions give us the distance from the fine pixel (third dimension) of i to all 
#######          the fine pixels of j.
    
    ####### We reshape dis_f to convert the last two dimensions iscale, iscale into one dimension iscale*iscale
    dis_f= np.reshape(dis_f,(N_c,N_c,iscale*iscale,iscale*iscale))
    
    
    # Now we define Gamma_ff
    # We model the variogram gamma_ff as a exponential 
    
    def Gamma_ff(pd_uni_c, sill, ran):
        
        Gamma_ff_temp=np.zeros(iscale*iscale)
        Gamma_cc=np.zeros((N_c,N_c))
        
        for i_coarse in range(N_c):
            for j_coarse in range(N_c):
                temp_var=0
                for i_fine in range(iscale*iscale):
                    for i in range(iscale*iscale):
                        Gamma_ff_temp[i] = sill * (1 - np.exp(-dis_f[i_coarse,j_coarse,i_fine,i]/(ran/3)))     # EXPONENTIAL
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
        
    sill=param_c[0]
    ran=param_c[1]
    
    sillran=np.array([sill,ran])
    
    ydata=Gamma_coarse
    xdata=pd_uni_c
    
    param, pcov= opt.curve_fit(Gamma_ff, xdata, ydata,sillran,method='lm')
    
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
                    Gamma_ff_temp[i] = param[0] * (1 - np.exp(-dis_f[i_coarse,j_coarse,i_fine,i]/(param[1]/3)))     # EXPONENTIAL
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
    
    Gamma_cc_r=Gamma_cc_m-Gamma_cc_m[0]
    
    # gamma_cc_regul compared to gamma_coarse    
    Diff = Gamma_coarse - Gamma_cc_r
    
    ####################### Gamma_fc estimation #########################
    
    Gamma_ff_temp=np.zeros(iscale*iscale)
    Gamma_fc=np.zeros((iscale*iscale,N_c))
    
    for j_coarse in range(N_c):
        temp_var=0
        for i_fine in range(iscale*iscale):
            for i in range(iscale*iscale):
                Gamma_ff_temp[i] = param[0] * (1 - np.exp(-dis_f[int(np.floor(0.5*block_size**2)),j_coarse,i_fine,i]/(param[1]/3)))     # EXPONENTIAL
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
    # for ic in range(b_radius,math.floor(cols/iscale)-b_radius):
    #     for ir in range(b_radius,math.floor(rows/iscale)-b_radius):
    #         for ir_2 in range((ir-b_radius)*iscale,(ir+b_radius)*iscale+iscale):
    #             for ic_2 in range((ic-b_radius)*iscale-2,(ic+b_radius)*iscale+iscale):
    #                 if mask_TT_unm[ir_2,ic_2]==0:
    #                     Delta_T_final[ir_2,ic_2] = 0
    #                 else:
    #                     temp_var = np.reshape(Delta_T[ir-b_radius:ir+b_radius+1,ic-b_radius:ic+b_radius+1],block_size**2) # We take the block of coarse pixels
    #                     # We multiply each coarse pixel in the block by the corresponding lambda and sum
    #                     Delta_T_final[ir_2,ic_2] = np.sum(lambdas[((ir_2-ir*iscale)*iscale+ic_2-ic*iscale)%lambdas.shape[0],:]*temp_var)
    
    
    for ic in range(b_radius,math.floor(cols/iscale)-b_radius):
        for ir in range(b_radius,math.floor(rows/iscale)-b_radius):
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

    # plt.figure()
    # plt.imshow(Temp)
    # plt.colorbar()
    # plt.savefig("tmp_fig/Temp.png")

    # plt.figure()
    # plt.imshow(TT_unm)
    # plt.colorbar()
    # plt.savefig("tmp_fig/TT_unm.png")

    # plt.figure()
    # plt.imshow(Delta_T_final)
    # plt.colorbar()
    # plt.savefig("tmp_fig/Delta_T_final.png")

    # plt.figure()
    # plt.imshow(TT_unmixed_corrected)
    # plt.colorbar()
    # plt.savefig("tmp_fig/TT_unmixed_corrected.png")
    # print("index",index.shape)
    # print("Temp",Temp.shape)
    return TT_unmixed_corrected, Delta_T_final, TT_unm

def setup_seed(seed):
    import random
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True




def get_ndvi_and_lst():

    in_file = '/content/drive/MyDrive/Ganglin/1_IMT/MCE_Projet3A/MODIS/MOD_2020_MOD11A1/tifs_files/1km_of_MOD13A2.A2020001.h18v04.061.2020326033640.hdf.0015.tif'
    red, nir, mir, cols, rows, projection, geotransform = read_tif_MOD13A2(in_file)
    # NVBI
    ndbi = (mir - nir) / (mir + nir)
    # SWIR 
    ndvi = (nir - red) / (nir + red)
    # 将NAN转化为0值
    nan_index = np.isnan(ndvi)
    ndvi[nan_index] = 0
    ndvi_1km = ndvi.astype(np.float32)

    nan_index = np.isnan(ndbi)
    ndbi[nan_index] = 0
    ndbi_1km = ndbi.astype(np.float32)

    in_file = '/content/drive/MyDrive/Ganglin/1_IMT/MCE_Projet3A/MODIS/MOD_2020_MOD11A1/tifs_files/2km_of_MOD13A2.A2020001.h18v04.061.2020326033640.hdf.0015.tif'
    red, nir, mir, cols, rows, projection, geotransform = read_tif_MOD13A2(in_file)
    # NDBI
    ndbi = (mir - nir) / (mir + nir)
    # NDVI 
    ndvi = (nir - red) / (nir + red)
    # Convert Nan to 0
    nan_index = np.isnan(ndvi)
    ndvi[nan_index] = 0
    ndvi_2km = ndvi.astype(np.float32)

    nan_index = np.isnan(ndbi)
    ndbi[nan_index] = 0
    ndbi_2km = ndbi.astype(np.float32)

    #
    tifs_1km_path = '/content/drive/MyDrive/Ganglin/1_IMT/MCE_Projet3A/MODIS/MOD_2020_MOD11A1/tifs_files/1km_of_MOD11A1.A2020001.h18v04.061.2021003092415.hdf.0015.tif'
    tifs_2km_path = '/content/drive/MyDrive/Ganglin/1_IMT/MCE_Projet3A/MODIS/MOD_2020_MOD11A1/tifs_files/2km_of_MOD11A1.A2020001.h18v04.061.2021003092415.hdf.0015.tif'
    LST_K_day_1km, LST_K_night_1km, cols_1km, rows_1km, projection_1km, geotransform_1km = read_tif(tifs_1km_path)
    LST_K_day_2km, LST_K_night_2km, cols_2km, rows_2km, projection_2km, geotransform_2km = read_tif(tifs_2km_path)

    return ndvi_1km, LST_K_day_1km, ndvi_2km, LST_K_day_2km




def get_mean_std(all_data_loader):
    mean_ndvi_1km   = []
    std_ndvi_1km    = []
    mean_ndvi_2km   = []
    std_ndvi_2km    = []

    mean_LST_K_day_1km   = []
    std_LST_K_day_1km    = []
    mean_LST_K_day_2km   = []
    std_LST_K_day_2km    = []
    for ndvi_1km, LST_K_day_1km, ndvi_2km, LST_K_day_2km in all_data_loader:
        mean_ndvi_1km.append(ndvi_1km.mean().item())
        mean_ndvi_2km.append(ndvi_2km.mean().item())
        mean_LST_K_day_1km.append(LST_K_day_1km.mean().item())
        mean_LST_K_day_2km.append(LST_K_day_2km.mean().item())

        std_ndvi_1km.append(ndvi_1km.std().item())
        std_ndvi_2km.append(ndvi_2km.std().item())
        std_LST_K_day_1km.append(LST_K_day_1km.std().item())
        std_LST_K_day_2km.append(LST_K_day_2km.std().item())


    mean_all = {}
    std_all = {}

    mean_all["ndvi_1km"] = np.mean(mean_ndvi_1km)
    mean_all["ndvi_2km"] = np.mean(mean_ndvi_2km)
    mean_all["LST_K_day_1km"] = np.mean(mean_LST_K_day_1km)
    mean_all["LST_K_day_2km"] = np.mean(mean_LST_K_day_2km)

    std_all["ndvi_1km"] = np.mean(std_ndvi_1km)
    std_all["ndvi_2km"] = np.mean(std_ndvi_2km)
    std_all["LST_K_day_1km"] = np.mean(std_LST_K_day_1km)
    std_all["LST_K_day_2km"] = np.mean(std_LST_K_day_2km)

    return mean_all, std_all
