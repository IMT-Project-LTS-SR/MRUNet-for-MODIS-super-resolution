def correction(path_index, path_temperature, path_unm, iscale, path_out, path_out1, path_out2, path_out3):
	
############## 
#
# This function takes three images (path_index, path_temperature and path_unm) and 
# computes the residuals of each coarse and fine pixel. For the coarse pixels 
# it computes the residuals as the difference between the measured coarse 
# temperature of the pixel and the mean of the fine pixel temperatures within 
# the coarse pixel $R_coarse=T_measured - mean(T_unm)$. Then the residuals for each 
# fine pixel resolution inside a coarse pixel is the residual of the coarse 
# pixel (scale invariance hypothesis).
#  	
############## Inputs/Outputs:
#
# path_index: String leading to the index image file. (.bsq, .img, .raw)
#             This file must contain only one band.
#
# path_temperature: String leading to the temperature image file. (.bsq, .img, .raw)	
#                   This file must contain only one band. Background pixels ==0
#
# path_unm: String leading to the unmixed temperature image file. (.bsq, .img, .raw)	
#           This file must contain only one band.
#
# iscale: size factor between measured pixel and unmixed pixel. 
#         Ex: if the initial resolution of Temp is 90m and the resolution 
#             after unmixing is 10m iscale=9. 
#
# path_out: String with the adress of the result image. 
#           path_out -- Delta_T 
#           path_out1 -- T_unm_add
#           path_out2 -- Delta_T_final
#           path_out3 -- T_unmixed_final
#
# Exemple:
# path_index = '/tmp_user/ldota214h/cgranero/CATUT_unmixing/VNIR-SWIR/4_Index-Estimation/080628/Index_080628_20m.bsq'
# path_temperature = '/tmp_user/ldota214h/cgranero/CATUT_unmixing/TIR/4_TES/Outputs_080628_20m/LST_4_bandes_jour_satellite_20m_bruite.bsq'
# path_unm = '/tmp_user/ldota214h/cgranero/CATUT_unmixing/TsHARP-DISTRAD/Scale_Inv/Step2_Images_080628/T_unm_20m_from40m_jourNDVI.bsq'
# iscale = 2
# path_out='/tmp_user/ldota214h/cgranero/CATUT_unmixing/TsHARP-DISTRAD/Scale_Inv/Step3_Images_080628/Delta_T_40m_from20m_jour'+ind_name+'.bsq'
# path_out1='/tmp_user/ldota214h/cgranero/CATUT_unmixing/TsHARP-DISTRAD/Scale_Inv/Step3_Images_080628/T_unm_add_40m_from20m_jour'+ind_name+'.bsq'
# path_out2='/tmp_user/ldota214h/cgranero/CATUT_unmixing/TsHARP-DISTRAD/Scale_Inv/Step3_Images_080628/Delta_T_final_20m_from40m_jour'+ind_name+'.bsq'
# path_out3='/tmp_user/ldota214h/cgranero/CATUT_unmixing/TsHARP-DISTRAD/Scale_Inv/Step3_Images_080628/T_unmixed_final_20m_from40m_jour'+ind_.bsq'
###
######
###########
############### C. Granero-Belinchon; ONERA, Toulouse; 01/2019

	import numpy as np
	from osgeo import gdal, gdalconst

####################################################################
################ LOADING DATA ######################################
####################################################################

################### INDEX ##########################################

# Reading index data
	filename_index = path_index
	dataset = gdal.Open(filename_index)

	cols = dataset.RasterXSize #Spatial dimension x
	rows = dataset.RasterYSize #Spatial dimension y

# Read as array the index
	index = dataset.ReadAsArray(0, 0, cols, rows).astype(np.float)

################### Temperature ######################################

# Reading temperature data
	TempFile = path_temperature
	dataset_Temp = gdal.Open(TempFile)

	cols_t = dataset_Temp.RasterXSize #Spatial dimension x
	rows_t = dataset_Temp.RasterYSize #Spatial dimension y

# Read as array
	Temp= dataset_Temp.ReadAsArray(0, 0, cols_t, rows_t).astype(np.float)
			
	Tm=Temp
	dt=dataset_Temp
	
################### UNMIXED Temperature ###########################		

#dataset
	T_unmix = gdal.Open(path_unm)

	cols = T_unmix.RasterXSize #Spatial dimension x
	rows = T_unmix.RasterYSize #Spatial dimension y

# Read as array
	TT_unm = T_unmix.ReadAsArray(0, 0, cols, rows).astype(np.float)
		
################# CORRECTION #######################################

# We define the matrix of added temperature		
	T_unm_add =np.zeros((int(np.floor(rows/iscale)),int(np.floor(cols/iscale))))														
		
# We define a mask in order to not use the background pixels. If a pixel of the finer scale image is zero, 
# the coarse pixel which contains this one is also zero. We have to choose this mask or the mask of line 188
	mask_TT_unm=np.zeros((rows,cols))
	indice_mask_TT=np.nonzero(TT_unm)
	mask_TT_unm[indice_mask_TT]=1

	for ic_sc in range(int(np.floor(cols/iscale))):
		for ir_sc in range(int(np.floor(rows/iscale))):
			zz=np.where(mask_TT_unm[ir_sc*iscale:(ir_sc*iscale+iscale),ic_sc*iscale:(ic_sc*iscale+iscale)]==0)
			lzz=np.array(len(zz[0]))
			if lzz >= (iscale**2)/2:
				T_unm_add[ir_sc,ic_sc] = 0
			else:
				T_unm_add[ir_sc,ic_sc] =  np.sum(TT_unm[ir_sc*iscale:(ir_sc*iscale+iscale),ic_sc*iscale:(ic_sc*iscale+iscale)])/(iscale**2-lzz)
						
# Because in the borders of the image we have background pixels, we have to dont take into account them. 
# To do that, we take the measured temperatures at 40m (respectively, 60m, 80m, 100m) and impose zero 
# values to all the pixels of the "Added temperature" that are zero for the measure temperature.		
	mask=np.greater(Tm,0) # We eliminate the background pixels (not in the image)
	for icol in range(int(np.floor(cols/iscale))):
		for irow in range(int(np.floor(rows/iscale))):
			if mask[irow,icol] == False:
				T_unm_add[irow,icol]=0
					
	mask2=np.greater(T_unm_add,0) # We eliminate the background pixels (not in the image)
	for icol in range(int(np.floor(cols/iscale))):
		for irow in range(int(np.floor(rows/iscale))):
			if mask2[irow,icol] == False:
				Tm[irow,icol]=0
	
# We define the delta at the scale of the measured temperature as in the model.		
			
																					
	Delta_T = Tm[0:int(np.floor(rows/iscale)),0:int(np.floor(cols/iscale))] -T_unm_add
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
###################################################################
########### SAVING RESULTS ########################################
###################################################################
# 1		
	driver = dt.GetDriver()
	outDs = driver.Create(path_out, int(np.floor(cols/iscale)), int(np.floor(rows/iscale)), 1, gdal.GDT_Float32)

	outBand = outDs.GetRasterBand(1)
	outBand.WriteArray(Delta_T, 0, 0)
	outBand.FlushCache()
# 2			
	driver = dt.GetDriver()
	outDs = driver.Create(path_out1, int(np.floor(cols/iscale)), int(np.floor(rows/iscale)), 1, gdal.GDT_Float32)

	outBand = outDs.GetRasterBand(1)
	outBand.WriteArray(T_unm_add, 0, 0)
	outBand.FlushCache()
# 3
	driver = dataset.GetDriver()
	outDs = driver.Create(path_out2, int(cols), int(rows), 1, gdal.GDT_Float32)

	outBand = outDs.GetRasterBand(1)
	outBand.WriteArray(Delta_T_final, 0, 0)
	outBand.FlushCache()
# 4
	driver = dataset.GetDriver()
	outDs = driver.Create(path_out3, int(cols), int(rows), 1, gdal.GDT_Float32)

	outBand = outDs.GetRasterBand(1)
	outBand.WriteArray(TT_unmixed_corrected, 0, 0)
	outBand.FlushCache()

	print('Correction Done')
	
	return


def correction_v2(path_index, path_fit, path_temperature, path_unm, iscale, path_out, path_out1, path_out2, path_out3):
	
############## 
#
# This function takes three images (path_index, path_temperature and path_unm) and 
# computes the residuals of each coarse and fine pixel. For the coarse pixels 
# it computes the residuals as the difference between the measured coarse 
# temperature of the pixel and the coarse temperature obtained with the linear regression 
# $R_coarse=T_measured_coarse - (a1*I_measured_coarse + a0)$. Then the residuals for each 
# fine pixel resolution inside a coarse pixel is the residual of the coarse 
# pixel (scale invariance hypothesis).
#  	
############## Inputs/Outputs:
#
# path_index: String leading to the index image file. (.bsq, .img, .raw)
#             This file must contain only one band.
#
# path_fit: String leading to the txt file with the regression parameters.
#
# path_temperature: String leading to the temperature image file. (.bsq, .img, .raw)	
#                   This file must contain only one band. Background pixels ==0
#
# path_unm: String leading to the unmixed temperature image file. (.bsq, .img, .raw)	
#           This file must contain only one band.
#
# iscale: size factor between measured pixel and unmixed pixel. 
#         Ex: if the initial resolution of Temp is 90m and the resolution 
#             after unmixing is 10m iscale=9. 
#
# path_out: String with the adress of the result image. 
#           path_out -- Delta_T 
#           path_out1 -- T_unm_add
#           path_out2 -- Delta_T_final
#           path_out3 -- T_unmixed_final
#
# Exemple:
# path_index = '/tmp_user/ldota214h/cgranero/CATUT_unmixing/VNIR-SWIR/4_Index-Estimation/080628/Index_080628_20m.bsq'
# path_fit = '/tmp_user/ldota214h/cgranero/CATUT_unmixing/TsHARP-DISTRAD/Scale_Inv/Fit2_NDVI_T_080628.txt'
# path_temperature = '/tmp_user/ldota214h/cgranero/CATUT_unmixing/TIR/4_TES/Outputs_080628_20m/LST_4_bandes_jour_satellite_20m_bruite.bsq'
# path_unm = '/tmp_user/ldota214h/cgranero/CATUT_unmixing/TsHARP-DISTRAD/Scale_Inv/Step2_Images_080628/T_unm_20m_from40m_jourNDVI.bsq'
# iscale = 2
# path_out='/tmp_user/ldota214h/cgranero/CATUT_unmixing/TsHARP-DISTRAD/Scale_Inv/Step3_Images_080628/Delta_T_40m_from20m_jour'+ind_name+'.bsq'
# path_out1='/tmp_user/ldota214h/cgranero/CATUT_unmixing/TsHARP-DISTRAD/Scale_Inv/Step3_Images_080628/T_unm_add_40m_from20m_jour'+ind_name+'.bsq'
# path_out2='/tmp_user/ldota214h/cgranero/CATUT_unmixing/TsHARP-DISTRAD/Scale_Inv/Step3_Images_080628/Delta_T_final_20m_from40m_jour'+ind_name+'.bsq'
# path_out3='/tmp_user/ldota214h/cgranero/CATUT_unmixing/TsHARP-DISTRAD/Scale_Inv/Step3_Images_080628/T_unmixed_final_20m_from40m_jour'+ind_.bsq'
###
######
###########
############### C. Granero-Belinchon; ONERA, Toulouse; 01/2019

	import numpy as np
	import numpy.ma as ma
	import matplotlib.pyplot as plt
	from osgeo import gdal, gdalconst
	import math
	from scipy.stats import linregress
	from scipy import stats


####################################################################
################ LOADING DATA ######################################
####################################################################

################### INDEX ########################################## # coarse scale

# Reading index data
	filename_index = path_index
	dataset = gdal.Open(filename_index)

	cols_i = dataset.RasterXSize #Spatial dimension x
	rows_i = dataset.RasterYSize #Spatial dimension y

# Read as array the index
	index = dataset.ReadAsArray(0, 0, cols_i, rows_i).astype(np.float)

################### Temperature ###################################### # coarse scale

# Reading temperature data
	TempFile = path_temperature
	dataset_Temp = gdal.Open(TempFile)

	cols_t = dataset_Temp.RasterXSize #Spatial dimension x
	rows_t = dataset_Temp.RasterYSize #Spatial dimension y

# Read as array
	Temp= dataset_Temp.ReadAsArray(0, 0, cols_t, rows_t).astype(np.float)
			
	Tm=Temp
	dt=dataset_Temp
	
################### UNMIXED Temperature ###########################	 # fine scale	

#dataset
	T_unmix = gdal.Open(path_unm)

	cols = T_unmix.RasterXSize #Spatial dimension x
	rows = T_unmix.RasterYSize #Spatial dimension y

# Read as array
	TT_unm = T_unmix.ReadAsArray(0, 0, cols, rows).astype(np.float)
		
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
		
# We define a mask in order to not use the background pixels. If a pixel of the finer scale image is zero, 
# the coarse pixel which contains this one is also zero. We have to choose this mask or the mask of line 188
	mask_TT_unm=np.zeros((rows,cols))
	indice_mask_TT=np.nonzero(TT_unm)
	mask_TT_unm[indice_mask_TT]=1
			
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
###################################################################
########### SAVING RESULTS ########################################
###################################################################
# 1		
	driver = dt.GetDriver()
	outDs = driver.Create(path_out, cols_t, rows_t, 1, gdal.GDT_Float32)

	outBand = outDs.GetRasterBand(1)
	outBand.WriteArray(Delta_T, 0, 0)
	outBand.FlushCache()
# 2			
	driver = dt.GetDriver()
	outDs = driver.Create(path_out1, cols_t, rows_t, 1, gdal.GDT_Float32)

	outBand = outDs.GetRasterBand(1)
	outBand.WriteArray(T_unm_add, 0, 0)
	outBand.FlushCache()
# 3
	driver = dataset.GetDriver()
	outDs = driver.Create(path_out2, int(cols), int(rows), 1, gdal.GDT_Float32)

	outBand = outDs.GetRasterBand(1)
	outBand.WriteArray(Delta_T_final, 0, 0)
	outBand.FlushCache()
# 4
	driver = dataset.GetDriver()
	outDs = driver.Create(path_out3, int(cols), int(rows), 1, gdal.GDT_Float32)

	outBand = outDs.GetRasterBand(1)
	outBand.WriteArray(TT_unmixed_corrected, 0, 0)
	outBand.FlushCache()

	print('Correction v2 Done')
	
	return
	
def quality_correction(path_index, path_temperature, path_unm, iscale, max_thre, min_thre, path_out, path_out1, path_out2, path_out3):
	
############## 
#
# This function takes three images (path_index, path_temperature and path_unm) and 
# computes the residuals of each coarse and fine pixel. For the coarse pixels 
# it computes the residuals as the difference between the measured coarse 
# temperature of the pixel and the mean of the fine pixel temperatures within 
# the coarse pixel $R_coarse=T_measured - mean(T_unm)$. Then the residuals for each 
# fine pixel resolution inside a coarse pixel is the residual of the coarse 
# pixel (scale invariance hypothesis).
#
# Before computing the residuals, this function look for pixels with temperature values
# above max_thre +5 or below min_thre - 5 and recalculates the temperature of these 
# pixels by averaging the temperatures of the 5x5 squares containing each non-desired 
# temperature pixel in the center.
#  	
############## Inputs/Outputs:
#
# path_index: String leading to the index image file. (.bsq, .img, .raw)
#             This file must contain only one band.
#
# path_temperature: String leading to the temperature image file. (.bsq, .img, .raw)	
#                   This file must contain only one band. Background pixels ==0
#
# path_unm: String leading to the unmixed temperature image file. (.bsq, .img, .raw)	
#           This file must contain only one band.
#
# iscale: size factor between measured pixel and unmixed pixel. 
#         Ex: if the initial resolution of Temp is 90m and the resolution 
#             after unmixing is 10m iscale=9. 
#
# max_thre: max_three+5 is the maximum accepted temperature value for the quality control. 
#           If == 0, then max_three == the maximal value of the measured 
#           temperature at the coarse resolution.
#
# min_thre: min_three-5 is the minimum accepted temperature value for the quality control.
#            If == 0, then min_three == the minimal value of the measured 
#           temperature at the coarse resolution.
# path_out: String with the adress of the result image. 
#           path_out -- Delta_T 
#           path_out1 -- T_unm_add
#           path_out2 -- Delta_T_final
#           path_out3 -- T_unmixed_final
#
# Exemple:
# path_index = '/tmp_user/ldota214h/cgranero/CATUT_unmixing/VNIR-SWIR/4_Index-Estimation/080628/Index_080628_20m.bsq'
# path_temperature = '/tmp_user/ldota214h/cgranero/CATUT_unmixing/TIR/4_TES/Outputs_080628_20m/LST_4_bandes_jour_satellite_20m_bruite.bsq'
# path_unm = '/tmp_user/ldota214h/cgranero/CATUT_unmixing/TsHARP-DISTRAD/Scale_Inv/Step2_Images_080628/T_unm_20m_from40m_jourNDVI.bsq'
# iscale = 2
# max_thre = 0
# min_thre = 0
# path_out='/tmp_user/ldota214h/cgranero/CATUT_unmixing/TsHARP-DISTRAD/Scale_Inv/Step3_Images_080628/Delta_T_40m_from20m_jour'+ind_name+'.bsq'
# path_out1='/tmp_user/ldota214h/cgranero/CATUT_unmixing/TsHARP-DISTRAD/Scale_Inv/Step3_Images_080628/T_unm_add_40m_from20m_jour'+ind_name+'.bsq'
# path_out2='/tmp_user/ldota214h/cgranero/CATUT_unmixing/TsHARP-DISTRAD/Scale_Inv/Step3_Images_080628/Delta_T_final_20m_from40m_jour'+ind_name+'.bsq'
# path_out3='/tmp_user/ldota214h/cgranero/CATUT_unmixing/TsHARP-DISTRAD/Scale_Inv/Step3_Images_080628/T_unmixed_final_20m_from40m_jour'+ind_.bsq'
###
######
###########
############### C. Granero-Belinchon; ONERA, Toulouse; 01/2019

	import numpy as np
	from osgeo import gdal, gdalconst
	import math

####################################################################
################ LOADING DATA ######################################
####################################################################

################### INDEX ##########################################

# Reading index data
	filename_index = path_index
	dataset = gdal.Open(filename_index)

	cols = dataset.RasterXSize #Spatial dimension x
	rows = dataset.RasterYSize #Spatial dimension y

# Read as array the index
	index = dataset.ReadAsArray(0, 0, cols, rows).astype(np.float)

################### Temperature ######################################

# Reading temperature data
	TempFile = path_temperature
	dataset_Temp = gdal.Open(TempFile)

	cols_t = dataset_Temp.RasterXSize #Spatial dimension x
	rows_t = dataset_Temp.RasterYSize #Spatial dimension y

# Read as array
	Temp= dataset_Temp.ReadAsArray(0, 0, cols_t, rows_t).astype(np.float)
			
	Tm=Temp
	dt=dataset_Temp
	
################### UNMIXED Temperature ###########################		

#dataset
	T_unmix = gdal.Open(path_unm)

	cols = T_unmix.RasterXSize #Spatial dimension x
	rows = T_unmix.RasterYSize #Spatial dimension y

# Read as array
	TT_unm = T_unmix.ReadAsArray(0, 0, cols, rows).astype(np.float)
	
##################### QUALITY CONTROL ##############################

	if max_thre == None:
	# We compute the max value of the measured temperature
		max_thre=np.amax(Tm)
		
	if min_thre == None:	
	# We compute the min value of the measured temperature
		Titos=Tm
		nn2=np.where(Titos==0)
		Titos[nn2]=1000	
		min_thre=np.amin(Titos)
		Titos[nn2]=0
	
	# We add all the distances of the 5x5 block of pixels.
	tot = 24 + 24*np.sqrt(2) + 16*np.sqrt(5)
		
	poids=np.zeros((5,5))
	for i in range(0,5):
		for j in range(0,5):
			poids[i,j]=2*np.sqrt((i-2)**2+(j-2)**2)
	poids=poids.flatten()
	
	# Quality control
	for icol in range(2,cols-2):
		for irow in range(2,rows-2):
			
			if TT_unm[irow,icol] > max_thre+5:
				
				zeros=np.zeros(25)
				vec=TT_unm[irow-2:irow+3,icol-2:icol+3]
				nn=np.nonzero(vec.flatten())
				zeros[nn]=1
				weight=poids*zeros
				
				TT_unm[irow,icol] = 1/np.sum(weight) * np.sum(poids*vec.flatten())
				
			if TT_unm[irow,icol] < min_thre-5 and TT_unm[irow,icol]!=0:
				
				zeros=np.zeros(25)
				vec=TT_unm[irow-2:irow+3,icol-2:icol+3]
				nn=np.nonzero(vec.flatten())
				zeros[nn]=1
				weight=poids*zeros
				
				TT_unm[irow,icol] = 1/np.sum(weight) * np.sum(poids*vec.flatten())
	
################# CORRECTION #######################################	
		
# We define the matrix of added temperature		
	T_unm_add =np.zeros((math.floor(rows/iscale),math.floor(cols/iscale)))														
		
# We define a mask in order to not use the background pixels. If a pixel of the finer scale image is zero, 
# the coarse pixel which contains this one is also zero. We have to choose this mask or the mask of line 188
	mask_TT_unm=np.zeros((rows,cols))
	indice_mask_TT=np.nonzero(TT_unm)
	mask_TT_unm[indice_mask_TT]=1

	for ic_sc in range(math.floor(cols/iscale)):
		for ir_sc in range(math.floor(rows/iscale)):
			zz=np.where(mask_TT_unm[ir_sc*iscale:(ir_sc*iscale+iscale),ic_sc*iscale:(ic_sc*iscale+iscale)]==0)
			lzz=np.array(len(zz[0]))
			if lzz >= (iscale**2)/2:
				T_unm_add[ir_sc,ic_sc] = 0
			else:
				T_unm_add[ir_sc,ic_sc] =  np.sum(TT_unm[ir_sc*iscale:(ir_sc*iscale+iscale),ic_sc*iscale:(ic_sc*iscale+iscale)])/(iscale**2-lzz)
						
# Because in the borders of the image we have background pixels, we have to dont take into account them. 
# To do that, we take the measured temperatures at 40m (respectively, 60m, 80m, 100m) and impose zero 
# values to all the pixels of the "Added temperature" that are zero for the measure temperature.		
	mask=np.greater(Tm,0) # We eliminate the background pixels (not in the image)
	for icol in range(math.floor(cols/iscale)):
		for irow in range(math.floor(rows/iscale)):
			if mask[irow,icol] == False:
				T_unm_add[irow,icol]=0
					
	mask2=np.greater(T_unm_add,0) # We eliminate the background pixels (not in the image)
	for icol in range(math.floor(cols/iscale)):
		for irow in range(math.floor(rows/iscale)):
			if mask2[irow,icol] == False:
				Tm[irow,icol]=0
	
# We define the delta at the scale of the measured temperature as in the model.		
			
																					
	Delta_T = Tm[0:math.floor(rows/iscale),0:math.floor(cols/iscale)] -T_unm_add
# We obtain the delta at the scale of the unmixed temperature
	Delta_T_final =np.zeros((rows,cols))
	for ic in range(math.floor(cols/iscale)):
		for ir in range(math.floor(rows/iscale)):
			for ir_2 in range(ir*iscale,(ir*iscale+iscale)):
				for ic_2 in range(ic*iscale,(ic*iscale+iscale)):
					if mask_TT_unm[ir_2,ic_2]==0:
						Delta_T_final[ir_2,ic_2] = 0
					else:
						Delta_T_final[ir_2,ic_2] = Delta_T[ir,ic]
		
# We correct the unmixed temperature using the delta	

	TT_unmixed_corrected = TT_unm + Delta_T_final
###################################################################
########### SAVING RESULTS ########################################
###################################################################
# 1		
	driver = dt.GetDriver()
	outDs = driver.Create(path_out, int(math.floor(cols/iscale)), int(math.floor(rows/iscale)), 1, gdal.GDT_Float32)

	outBand = outDs.GetRasterBand(1)
	outBand.WriteArray(Delta_T, 0, 0)
	outBand.FlushCache()
# 2			
	driver = dt.GetDriver()
	outDs = driver.Create(path_out1, int(math.floor(cols/iscale)), int(math.floor(rows/iscale)), 1, gdal.GDT_Float32)

	outBand = outDs.GetRasterBand(1)
	outBand.WriteArray(T_unm_add, 0, 0)
	outBand.FlushCache()
# 3
	driver = dataset.GetDriver()
	outDs = driver.Create(path_out2, int(cols), int(rows), 1, gdal.GDT_Float32)

	outBand = outDs.GetRasterBand(1)
	outBand.WriteArray(Delta_T_final, 0, 0)
	outBand.FlushCache()
# 4
	driver = dataset.GetDriver()
	outDs = driver.Create(path_out3, int(cols), int(rows), 1, gdal.GDT_Float32)

	outBand = outDs.GetRasterBand(1)
	outBand.WriteArray(TT_unmixed_corrected, 0, 0)
	outBand.FlushCache()

	print('Correction Done')
	
	return
	
	
def correction_ATPRK(path_index, path_fit, path_temperature, path_unm, iscale, scc, block_size, sill, ran, path_out, path_out1, path_out2, path_out3,path_plot):
	
############## 
#
# This function takes three images (path_index, path_temperature and path_unm) and 
# computes the residuals of each coarse and fine pixel. For the coarse pixels 
# it computes the residuals as the difference between the measured coarse 
# temperature of the pixel and the coarse temperature obtained with the linear regression 
# $R_coarse=T_measured_coarse - (a1*I_measured_coarse + a0)$. Then the residuals for each 
# fine pixel resolution inside a coarse pixel is computed using the ATPK procedure explained
# in Wang, Shi, Atkinson 2016, 'Area-to-point regression kriging for pan-sharpening'.
#  	
############## Inputs/Outputs:
#
# path_index: String leading to the index image file. (.bsq, .img, .raw)
#             This file must contain only one band.
#
# path_fit: String leading to the txt file with the regression parameters.
#
# path_temperature: String leading to the temperature image file. (.bsq, .img, .raw)	
#                   This file must contain only one band. Background pixels ==0
#
# path_unm: String leading to the unmixed temperature image file. (.bsq, .img, .raw)	
#           This file must contain only one band.
#
# iscale: size factor between measured pixel and unmixed pixel. 
#         Ex: if the initial resolution of Temp is 90m and the resolution 
#             after unmixing is 10m iscale=9. 
#
# scc: The coarse pixel has a size scc * scc
#
# block_size: The moving window used to compute the semivariograms and the weights in the 
#             ATPK procedures has a size block_size * block_size 
#
# sill and ran are the parameters defining the shape of the semivariogram. see line 199
#
# path_out: String with the adress of the result image. 
#           path_out -- Delta_T 
#           path_out1 -- T_unm_add
#           path_out2 -- Delta_T_final
#           path_out3 -- T_unmixed_final
#
# path_plot: String with the adress of the fit image of the coarse semivariogram.
# 			 The fit must be good for the results to be good.
#
# Exemple:
# path_index = '/tmp_user/ldota214h/cgranero/CATUT_unmixing/VNIR-SWIR/4_Index-Estimation/080628/Index_080628_40m.bsq'
# path_fit = '/tmp_user/ldota214h/cgranero/CATUT_unmixing/TsHARP-DISTRAD/Scale_Inv/Fit2_NDVI_T_080628.txt'
# path_temperature = '/tmp_user/ldota214h/cgranero/CATUT_unmixing/TIR/4_TES/Outputs_080628_40m/LST_4_bandes_jour_satellite_40m_bruite.bsq'
# path_unm = '/tmp_user/ldota214h/cgranero/CATUT_unmixing/TsHARP-DISTRAD/Scale_Inv/Step2_Images_080628/T_unm_20m_from40m_jourNDVI.bsq'
# iscale = 2
# scc = 40
# block_size = 5
# sill=7
# ran=1000
# path_out='/tmp_user/ldota214h/cgranero/CATUT_unmixing/TsHARP-DISTRAD/Scale_Inv/Step3_Images_080628/Delta_T_40m_from20m_jour'+ind_name+'.bsq'
# path_out1='/tmp_user/ldota214h/cgranero/CATUT_unmixing/TsHARP-DISTRAD/Scale_Inv/Step3_Images_080628/T_unm_add_40m_from20m_jour'+ind_name+'.bsq'
# path_out2='/tmp_user/ldota214h/cgranero/CATUT_unmixing/TsHARP-DISTRAD/Scale_Inv/Step3_Images_080628/Delta_T_final_20m_from40m_jour'+ind_name+'.bsq'
# path_out3='/tmp_user/ldota214h/cgranero/CATUT_unmixing/TsHARP-DISTRAD/Scale_Inv/Step3_Images_080628/T_unmixed_final_20m_from40m_jour'+ind_.bsq'
# path_plot= '/tmp_user/ldota214h/cgranero/CATUT_unmixing/ATPRK/Scale_Inv/Fit_Plots_080628/Gamma_c_NDVI_40m.png'
###
######
###########
############### C. Granero-Belinchon; ONERA, Toulouse; 01/2019

	import numpy as np
	import matplotlib.pyplot as plt
	from osgeo import gdal, gdalconst
	import math
	from scipy.spatial.distance import pdist, squareform
	import scipy.optimize as opt	

####################################################################
################ LOADING DATA ######################################
####################################################################

	b_radius = int(np.floor(block_size/2))
	
################### INDEX Coarse ###################################

# Reading index data
	filename_index = path_index
	dataset = gdal.Open(filename_index)

	cols_i = dataset.RasterXSize #Spatial dimension x
	rows_i = dataset.RasterYSize #Spatial dimension y

# Read as array the index
	index = dataset.ReadAsArray(0, 0, cols_i, rows_i).astype(np.float)

################### Temperature Coarse #############################

# Reading temperature data
	TempFile = path_temperature
	dataset_Temp = gdal.Open(TempFile)

	cols_t = dataset_Temp.RasterXSize #Spatial dimension x
	rows_t = dataset_Temp.RasterYSize #Spatial dimension y

# Read as array
	Temp= dataset_Temp.ReadAsArray(0, 0, cols_t, rows_t).astype(np.float)
			
	Tm=Temp
	dtset=dataset_Temp
	
################### UNMIXED Temperature ###########################		

#dataset
	T_unmix = gdal.Open(path_unm)

	cols = T_unmix.RasterXSize #Spatial dimension x
	rows = T_unmix.RasterYSize #Spatial dimension y

# Read as array
	TT_unm = T_unmix.ReadAsArray(0, 0, cols, rows).astype(np.float)
		
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
					Gamma_coarse_temp[irow,icol,idist]	= Gamma_coarse_temp[irow,icol,idist]/(2*ii)		
	
	#Gamma_coarse_temp[np.isnan(Gamma_coarse_temp)]=0	
	
	Gamma_coarse=np.zeros(len(distances))
	for idist in range(len(pd_uni_c)):
		zz=np.nonzero(Gamma_coarse_temp[:,:,idist])
		gg=Gamma_coarse_temp[zz[0],zz[1],idist]
		Gamma_coarse[idist]=np.mean(gg)
	
	Gamma_coarse[np.isnan(Gamma_coarse)]=0	
	
	def Func_Gamma_cc(pd_uni_c, sill, ran):
		
		Gamma_c_temp = sill * (1 - np.exp(-pd_uni_c/(ran/3))) 	# EXPONENTIAL
		
		return Gamma_c_temp
	
	sillran_c=np.array([sill,ran])
	
	ydata=Gamma_coarse
	xdata=pd_uni_c
	
	param_c, pcov_c = opt.curve_fit(Func_Gamma_cc, xdata, ydata,sillran_c,method='lm')
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

###################################################################
########### SAVING RESULTS ########################################
###################################################################
# 1		
	driver = dtset.GetDriver()
	outDs = driver.Create(path_out, cols_t, rows_t, 1, gdal.GDT_Float32)

	outBand = outDs.GetRasterBand(1)
	outBand.WriteArray(Delta_T, 0, 0)
	outBand.FlushCache()
# 2			
	driver = dtset.GetDriver()
	outDs = driver.Create(path_out1,cols_t, rows_t, 1, gdal.GDT_Float32)

	outBand = outDs.GetRasterBand(1)
	outBand.WriteArray(T_unm_add, 0, 0)
	outBand.FlushCache()
# 3
	driver = dataset.GetDriver()
	outDs = driver.Create(path_out2, int(cols), int(rows), 1, gdal.GDT_Float32)

	outBand = outDs.GetRasterBand(1)
	outBand.WriteArray(Delta_T_final, 0, 0)
	outBand.FlushCache()
# 4
	driver = dataset.GetDriver()
	outDs = driver.Create(path_out3, int(cols), int(rows), 1, gdal.GDT_Float32)

	outBand = outDs.GetRasterBand(1)
	outBand.WriteArray(TT_unmixed_corrected, 0, 0)
	outBand.FlushCache()

	print('Correction ATPK Done')
	
	return	
	
def correction_AATPRK(path_index, path_slope, path_intercept, path_temperature, path_unm, iscale, scc, block_size, sill, ran, path_out, path_out1, path_out2, path_out3,path_plot):
	
############## 
#
# This function takes three images (path_index, path_temperature and path_unm) and 
# computes the residuals of each coarse and fine pixel. For the coarse pixels 
# it computes the residuals as the difference between the measured coarse 
# temperature of the pixel and the coarse temperature obtained with the linear regression 
# $R_coarse=T_measured_coarse - (a1*I_measured_coarse + a0)$. Then the residuals for each 
# fine pixel resolution inside a coarse pixel is computed using the ATPK procedure explained
# in Wang, Shi, Atkinson 2016, 'Area-to-point regression kriging for pan-sharpening'.
#  	
############## Inputs/Outputs:
#
# path_index: String leading to the index image file. (.bsq, .img, .raw)
#             This file must contain only one band.
#
# path_intercept='/tmp_user/ldota214h/cgranero/CATUT_unmixing/AATPRK/Scale_Inv/Step1_Fits_'+date+'/Intercept_40m_NDVI.bsq' 
#
# path_slope='/tmp_user/ldota214h/cgranero/CATUT_unmixing/AATPRK/Scale_Inv/Step1_Fits_'+date+'/Slope_40m_NDVI.bsq'
#
# path_temperature: String leading to the temperature image file. (.bsq, .img, .raw)	
#                   This file must contain only one band. Background pixels ==0
#
# path_unm: String leading to the unmixed temperature image file. (.bsq, .img, .raw)	
#           This file must contain only one band.
#
# iscale: size factor between measured pixel and unmixed pixel. 
#         Ex: if the initial resolution of Temp is 90m and the resolution 
#             after unmixing is 10m iscale=9. 
#
# scc: The coarse pixel has a size scc * scc
#
# block_size: The moving window used to compute the semivariograms and the weights in the 
#             ATPK procedures has a size block_size * block_size 
#
# sill and ran are the parameters defining the shape of the semivariogram. see line 199
#
# path_out: String with the adress of the result image. 
#           path_out -- Delta_T 
#           path_out1 -- T_unm_add
#           path_out2 -- Delta_T_final
#           path_out3 -- T_unmixed_final
#
# path_plot: String with the adress of the fit image of the coarse semivariogram.
# 			 The fit must be good for the results to be good.
#
# Exemple:
# path_index = '/tmp_user/ldota214h/cgranero/CATUT_unmixing/VNIR-SWIR/4_Index-Estimation/080628/Index_080628_40m.bsq'
# path_fit = '/tmp_user/ldota214h/cgranero/CATUT_unmixing/TsHARP-DISTRAD/Scale_Inv/Fit2_NDVI_T_080628.txt'
# path_temperature = '/tmp_user/ldota214h/cgranero/CATUT_unmixing/TIR/4_TES/Outputs_080628_40m/LST_4_bandes_jour_satellite_40m_bruite.bsq'
# path_unm = '/tmp_user/ldota214h/cgranero/CATUT_unmixing/TsHARP-DISTRAD/Scale_Inv/Step2_Images_080628/T_unm_20m_from40m_jourNDVI.bsq'
# iscale = 2
# scc = 40
# block_size = 5
# sill=7
# ran=1000
# path_out='/tmp_user/ldota214h/cgranero/CATUT_unmixing/TsHARP-DISTRAD/Scale_Inv/Step3_Images_080628/Delta_T_40m_from20m_jour'+ind_name+'.bsq'
# path_out1='/tmp_user/ldota214h/cgranero/CATUT_unmixing/TsHARP-DISTRAD/Scale_Inv/Step3_Images_080628/T_unm_add_40m_from20m_jour'+ind_name+'.bsq'
# path_out2='/tmp_user/ldota214h/cgranero/CATUT_unmixing/TsHARP-DISTRAD/Scale_Inv/Step3_Images_080628/Delta_T_final_20m_from40m_jour'+ind_name+'.bsq'
# path_out3='/tmp_user/ldota214h/cgranero/CATUT_unmixing/TsHARP-DISTRAD/Scale_Inv/Step3_Images_080628/T_unmixed_final_20m_from40m_jour'+ind_.bsq'
# path_plot= '/tmp_user/ldota214h/cgranero/CATUT_unmixing/ATPRK/Scale_Inv/Fit_Plots_080628/Gamma_c_NDVI_40m.png'
###
######
###########
############### C. Granero-Belinchon; ONERA, Toulouse; 01/2019

	import numpy as np
	import matplotlib.pyplot as plt
	from osgeo import gdal, gdalconst
	import math
	from scipy.spatial.distance import pdist, squareform
	import scipy.optimize as opt	

####################################################################
################ LOADING DATA ######################################
####################################################################

	b_radius = int(np.floor(block_size/2))
	
################### INDEX Coarse ###################################

# Reading index data
	filename_index = path_index
	dataset = gdal.Open(filename_index)

	cols_i = dataset.RasterXSize #Spatial dimension x
	rows_i = dataset.RasterYSize #Spatial dimension y

# Read as array the index
	index = dataset.ReadAsArray(0, 0, cols_i, rows_i).astype(np.float)

################### Temperature Coarse #############################

# Reading temperature data
	TempFile = path_temperature
	dataset_Temp = gdal.Open(TempFile)

	cols_t = dataset_Temp.RasterXSize #Spatial dimension x
	rows_t = dataset_Temp.RasterYSize #Spatial dimension y

# Read as array
	Temp= dataset_Temp.ReadAsArray(0, 0, cols_t, rows_t).astype(np.float)
			
	Tm=Temp
	dtset=dataset_Temp
	
################### UNMIXED Temperature ###########################		

#dataset
	T_unmix = gdal.Open(path_unm)

	cols = T_unmix.RasterXSize #Spatial dimension x
	rows = T_unmix.RasterYSize #Spatial dimension y

# Read as array
	TT_unm = T_unmix.ReadAsArray(0, 0, cols, rows).astype(np.float)
		
################### Intercept #####################################

# Reading of data
	intercept_file = path_intercept

#datasets
	intercept_dataset = gdal.Open(intercept_file)

	cols_int = intercept_dataset.RasterXSize #Spatial dimension x
	rows_int = intercept_dataset.RasterYSize #Spatial dimension y

# Read as array
	intercept = intercept_dataset.ReadAsArray(0, 0, cols_int, rows_int).astype(np.float)


################### Slope #####################################

# Reading of data
	slope_file = path_slope

#datasets
	slope_dataset = gdal.Open(slope_file)

	cols_slope = slope_dataset.RasterXSize #Spatial dimension x
	rows_slope = slope_dataset.RasterYSize #Spatial dimension y

# Read as array
	slope = slope_dataset.ReadAsArray(0, 0, cols_slope, rows_slope).astype(np.float)
	
####
	
	mask=np.greater(Tm,0) # We eliminate the background pixels (not in the image)
	
	T_unm_add =np.zeros((rows_t,cols_t))
	
	for i in range(rows_t):
		for j in range(cols_t):
			T_unm_add[i,j] = intercept[i,j] + slope[i,j] * index[i,j]
			if mask[i,j] == False:
				T_unm_add[i,j]=0

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
					Gamma_coarse_temp[irow,icol,idist]	= Gamma_coarse_temp[irow,icol,idist]/(2*ii)		
	
	#Gamma_coarse_temp[np.isnan(Gamma_coarse_temp)]=0	
	
	Gamma_coarse=np.zeros(len(distances))
	for idist in range(len(pd_uni_c)):
		zz=np.nonzero(Gamma_coarse_temp[:,:,idist])
		gg=Gamma_coarse_temp[zz[0],zz[1],idist]
		Gamma_coarse[idist]=np.mean(gg)
	
	Gamma_coarse[np.isnan(Gamma_coarse)]=0	
	
	def Func_Gamma_cc(pd_uni_c, sill, ran):
		
		Gamma_c_temp = sill * (1 - np.exp(-pd_uni_c/(ran/3))) 	# EXPONENTIAL
		
		return Gamma_c_temp
	
	sillran_c=np.array([sill,ran])
	
	ydata=Gamma_coarse
	xdata=pd_uni_c
	
	param_c, pcov_c = opt.curve_fit(Func_Gamma_cc, xdata, ydata,sillran_c,method='lm')
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

###################################################################
########### SAVING RESULTS ########################################
###################################################################
# 1		
	driver = dtset.GetDriver()
	outDs = driver.Create(path_out, cols_t, rows_t, 1, gdal.GDT_Float32)

	outBand = outDs.GetRasterBand(1)
	outBand.WriteArray(Delta_T, 0, 0)
	outBand.FlushCache()
# 2			
	driver = dtset.GetDriver()
	outDs = driver.Create(path_out1, cols_t, rows_t, 1, gdal.GDT_Float32)

	outBand = outDs.GetRasterBand(1)
	outBand.WriteArray(T_unm_add, 0, 0)
	outBand.FlushCache()
# 3
	driver = dataset.GetDriver()
	outDs = driver.Create(path_out2, int(cols), int(rows), 1, gdal.GDT_Float32)

	outBand = outDs.GetRasterBand(1)
	outBand.WriteArray(Delta_T_final, 0, 0)
	outBand.FlushCache()
# 4
	driver = dataset.GetDriver()
	outDs = driver.Create(path_out3, int(cols), int(rows), 1, gdal.GDT_Float32)

	outBand = outDs.GetRasterBand(1)
	outBand.WriteArray(TT_unmixed_corrected, 0, 0)
	outBand.FlushCache()

	print('Correction ATPK Done')
	
	return
