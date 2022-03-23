def linear_unmixing(path_index, path_temperature, path_fit, iscale, path_mask, path_out):
	
############## 
#
# This function takes two images (path_index and path_temperature) and the linear regression 
# parameters on the text file (path_fit) and applies $T = a1*I +a0$, where T is the unknown
# a1 and a0 are the regression parameter and I is the index image.
#  	
############## Inputs/Outputs:
#
# path_index: String leading to the index image file. (.bsq, .img, .raw)
#             This file must contain only one band.
#
# path_temperature: String leading to the index image file. (.bsq, .img, .raw)	
#                   This file must contain only one band. the background pixels
#                   must be == 0
#
# path_fit: String leading to the txt file with the regression parameters.
#
# path_out: String with the adress where saving the image of T unmixed.
#
# iscale: Ratio between coarse and fine scale. For coarse =40m and fine =20 iscale =2
#
# path_mask: String leading to mask image to put the temperature values 
#            at zero where mask==0. If path_mask=0 then the mask is built in 
#            this script using the coarse temperature.
#
# Exemple:
# path_index = '/tmp_user/ldota214h/cgranero/CATUT_unmixing/VNIR-SWIR/4_Index-Estimation/080628/Index_080628_20m.bsq'
# path_temperature = '/tmp_user/ldota214h/cgranero/CATUT_unmixing/TIR/4_TES/Outputs_080628_20m/LST_4_bandes_jour_satellite_40m_bruite.bsq'
# path_fit = '/tmp_user/ldota214h/cgranero/CATUT_unmixing/TsHARP-DISTRAD/Scale_Inv/Fit2_NDVI_T_080628.txt'
# path_out = '/tmp_user/ldota214h/cgranero/CATUT_unmixing/TsHARP-DISTRAD/Scale_Inv/Step2_Images_080628/T_unm_20m_from40m_jourNDVI.bsq'
# path_mask = 0
# iscale=2 
###
######
###########
############### C. Granero-Belinchon; ONERA, Toulouse; 01/2019

	import numpy as np
	from osgeo import gdal, gdalconst
	
####################################################################
################ LOADING DATA ######################################
####################################################################

################### INDEX ########################################## # fine scale

# Reading index data
	filename_index = path_index
	dataset = gdal.Open(filename_index)

	cols = dataset.RasterXSize #Spatial dimension x
	rows = dataset.RasterYSize #Spatial dimension y

# Read as array the index
	index = dataset.ReadAsArray(0, 0, cols, rows).astype(np.float)

################### Temperature ###################################### # coarse scale

# Reading temperature data
	TempFile = path_temperature
	dataset_Temp = gdal.Open(TempFile)

	cols_t = dataset_Temp.RasterXSize #Spatial dimension x
	rows_t = dataset_Temp.RasterYSize #Spatial dimension y

# Read as array
	Temp= dataset_Temp.ReadAsArray(0, 0, cols_t, rows_t).astype(np.float)
	
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
	#	for irow in range(rows):
	#		if I_mask[irow,icol] == False:
	#			T_unm[irow,icol]=0
		
###################################################################
########### SAVING RESULTS ########################################
###################################################################

	driver = dataset_Temp.GetDriver()
	outDs = driver.Create(path_out, int(cols), int(rows), 1, gdal.GDT_Float32)

	outBand = outDs.GetRasterBand(1)
	outBand.WriteArray(T_unm, 0, 0)
	outBand.FlushCache()

	print('Unmixing Done')
	
	return
	
def bilinear_unmixing(path_index1, path_index2, path_temperature, path_fit, iscale, path_mask, path_out):
	
############## 
#
# This function takes three images (path_index1, path_index2 and path_temperature) and 
# the bilinear regression parameters on the text file (path_fit) and applies 
# $T = a1*I1 +a2*I2 +a0$, where T is the unknown, a2, a1 and a0 are the regression 
# parameters and I1 and I2 are the index images.
#  	
############## Inputs/Outputs:
#
# path_index: String leading to the index image file. (.bsq, .img, .raw)
#             This file must contain only one band.
#
# path_temperature: String leading to the index image file. (.bsq, .img, .raw)	
#                   This file must contain only one band. The Background pixels 
#                   must be == 0 
#
# path_fit: String leading to the txt file with the regression parameters.
#
# iscale: Ratio between coarse and fine scale. For coarse =40m and fine =20 iscale =2
#
# path_mask: String leading to mask image to put the temperature values 
#            at zero where mask==0. If path_mask=0 then the mask is built in 
#            this script using the coarse temperature.
#
# path_out: String with the adress where saving the image of T unmixed.
#
# Exemple:
# path_index = '/tmp_user/ldota214h/cgranero/CATUT_unmixing/VNIR-SWIR/4_Index-Estimation/080628/Index_080628_20m.bsq'
# path_temperature = '/tmp_user/ldota214h/cgranero/CATUT_unmixing/TIR/4_TES/Outputs_080628_20m/LST_4_bandes_jour_satellite_20m_bruite.bsq'
# path_fit = '/tmp_user/ldota214h/cgranero/CATUT_unmixing/TsHARP-DISTRAD/Scale_Inv/Fit2_NDVI_T_080628.txt'
# path_out = '/tmp_user/ldota214h/cgranero/CATUT_unmixing/TsHARP-DISTRAD/Scale_Inv/Step2_Images_080628/T_unm_20m_from40m_jourNDVI.bsq'
# path_mask = 0
# iscale=2 

###
######
###########
############### C. Granero-Belinchon; ONERA, Toulouse; 01/2019

	import numpy as np
	from osgeo import gdal, gdalconst

####################################################################
################ LOADING DATA ######################################
####################################################################

################### INDEX 1 ########################################## # fine scale

# Reading index data
	filename_index1 = path_index1
	dataset1 = gdal.Open(filename_index1)

	cols1 = dataset1.RasterXSize #Spatial dimension x
	rows1 = dataset1.RasterYSize #Spatial dimension y

# Read as array the index
	index1 = dataset1.ReadAsArray(0, 0, cols1, rows1).astype(np.float)
	
################### INDEX 2 ########################################## # fine scale

# Reading index data
	filename_index2 = path_index2
	dataset2 = gdal.Open(filename_index2)

	cols2 = dataset2.RasterXSize #Spatial dimension x
	rows2 = dataset2.RasterYSize #Spatial dimension y

# Read as array the index
	index2 = dataset2.ReadAsArray(0, 0, cols2, rows2).astype(np.float)

################### Temperature ###################################### # coarse scale

# Reading temperature data
	TempFile = path_temperature
	dataset_Temp = gdal.Open(TempFile)

	cols_t = dataset_Temp.RasterXSize #Spatial dimension x
	rows_t = dataset_Temp.RasterYSize #Spatial dimension y

# Read as array
	Temp= dataset_Temp.ReadAsArray(0, 0, cols_t, rows_t).astype(np.float)
	
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
	
## Param[0] indicates that we call the intercept of the linear regression.
	p0 = Param[0]
## Param[1] (Param[2]) indicate that we call the slopes of the linear regression.
	p1 = Param[1]
	p2 = Param[2]
		
	T_unm = p0 + p1*index1 + p2*index2
	
	if path_mask == 0:

		for ir in range(int(np.floor(rows1/iscale))):
			for ic in range(int(np.floor(cols1/iscale))):
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
	#	for irow in range(rows):
	#		if I_mask[irow,icol] == False:
	#			T_unm[irow,icol]=0
		
###################################################################
########### SAVING RESULTS ########################################
###################################################################

	driver = dataset_Temp.GetDriver()
	outDs = driver.Create(path_out, int(cols1), int(rows1), 1, gdal.GDT_Float32)

	outBand = outDs.GetRasterBand(1)
	outBand.WriteArray(T_unm, 0, 0)
	outBand.FlushCache()

	print('Unmixing Done')
	
	return

def linear_unmixing_byclass(path_index, path_temperature, path_class, path_fit, iscale, path_mask, path_out):
	
############## 
#
# This function takes three images (path_index, path_class and path_temperature) and 
# the linear regression parameters for each class defined in path_class on the 
# text file (path_fit) and applies $T = a1*I +a0$, where T is the unknown, a1 and 
# a0 are the regression parameters (different for each class) and I the index image.
#  	
############## Inputs/Outputs:
#
# path_index: String leading to the index image file. (.bsq, .img, .raw)
#             This file must contain only one band.
#
# path_temperature: String leading to the index image file. (.bsq, .img, .raw)	
#                   This file must contain only one band. the background pixels
#                   must be == 0
#
# path_class : String leading to the classification image file. (.bsq, .img, .raw)
#			   This file must contain only one band. background pixels =0
#
# path_fit: String leading to the txt file with the regression parameters. 
#			The same path using in Fit_ByClass. the code modify this path 
#			to call the different classes.
#
# iscale: Ratio between coarse and fine scale. For coarse =40m and fine =20 iscale =2
#
# path_mask: String leading to mask image to put the temperature values 
#            at zero where mask==0. If path_mask=0 then the mask is built in 
#            this script using the coarse temperature.
#
# path_out: String with the adress where saving the image of T unmixed.
#
# Exemple:
# path_index = '/tmp_user/ldota214h/cgranero/CATUT_unmixing/VNIR-SWIR/4_Index-Estimation/080628/Index_080628_20m.bsq'
# path_temperature = '/tmp_user/ldota214h/cgranero/CATUT_unmixing/TIR/4_TES/Outputs_080628_20m/LST_4_bandes_jour_satellite_20m_bruite.bsq'
# path_class = '/tmp_user/ldota214h/cgranero/CATUT_unmixing/Classif_Unmixing/Index_Based_Classification/Classif_080628/Classif_20m.bsq' 
# path_fit = '/tmp_user/ldota214h/cgranero/CATUT_unmixing/TsHARP-DISTRAD/Scale_Inv/Fit2_NDVI_T_080628.txt'
# iscale=2
# path_mask = 0
# path_out = '/tmp_user/ldota214h/cgranero/CATUT_unmixing/TsHARP-DISTRAD/Scale_Inv/Step2_Images_080628/T_unm_20m_from40m_jourNDVI.bsq'
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
	
################### Mask ###################################### # fine scale

	if path_mask !=0:
# Reading mask data
		MaskFile = path_mask
		dataset_Mask = gdal.Open(MaskFile)

		cols_m = dataset_Mask.RasterXSize #Spatial dimension x
		rows_m = dataset_Mask.RasterYSize #Spatial dimension y

# Read as array
		mask= dataset_Mask.ReadAsArray(0, 0, cols_m, rows_m).astype(np.float)
	
################# Classification #####################################

# Reading of data
	ClassFile = path_class

#datasets
	dataset_Class = gdal.Open(ClassFile)

	cols = dataset_Class.RasterXSize #Spatial dimension x
	rows = dataset_Class.RasterYSize #Spatial dimension y

# Read as array
	Class = dataset_Class.ReadAsArray(0, 0, cols, rows).astype(np.float)

# We obtain the number of classes and the value characterizing each class
	val_class = np.unique(Class)
	val_class = val_class[np.nonzero(val_class)]
	num_class = len(val_class)

###################################################################
########### UNMIXING ##############################################
###################################################################
	
	# We initialize the image result
	T_unm=np.zeros((rows,cols))
	
	for i in range(num_class):
	
		path_text = path_fit[:-4] + "_class"+ str(i+1) + path_fit[-4:]
		Param = np.genfromtxt(path_text)
	
		## Param[1] indicates that we call the intercept of the linear regression.
		a0 = Param[1]
		## Param[0] indicates that we call the slope of the linear regression.
		a1 = Param[0]
	
		for icol in range(cols_t):
			for irow in range(rows_t):
				if Class[irow,icol] == val_class[i]:
					T_unm[irow,icol] = a0 + a1*index[irow,icol]
	
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
	#	for irow in range(rows):
	#		if I_mask[irow,icol] == False:
	#			T_unm[irow,icol]=0
		
###################################################################
########### SAVING RESULTS ########################################
###################################################################

	driver = dataset_Temp.GetDriver()
	outDs = driver.Create(path_out, cols, rows, 1, gdal.GDT_Float32)

	outBand = outDs.GetRasterBand(1)
	outBand.WriteArray(T_unm, 0, 0)
	outBand.FlushCache()

	print('Unmixing Done')
	
	return

def huts_unmixing(path_index, path_albedo, path_temperature, path_fit, iscale, path_mask, path_out):
	
############## 
#
# This function takes three images (path_index, path_albedo and path_temperature) and 
# the HUTS regression parameters on the text file (path_fit) and applies 
# the HUTS formula, where T is the unknown, p0, p1 ... p14 are the regression 
# parameters and I and \alpha are the index and albedo images respectively.
#  	
############## Inputs/Outputs::
#
# path_index: String leading to the index image file. (.bsq, .img, .raw)
#             This file must contain only one band.
#
# path_albedo: String leading to the albedo image file. (.bsq, .img, .raw)
#              This file must contain only one band.
#
# path_temperature: String leading to the index image file. (.bsq, .img, .raw)	
#                   This file must contain only one band. The Background pixels 
#                   must be == 0 
#
# path_fit: String leading to the txt file with the regression parameters.
#
# iscale: Ratio between coarse and fine scale. For coarse =40m and fine =20 iscale =2
#
# path_mask: String leading to mask image to put the temperature values 
#            at zero where mask==0. If path_mask=0 then the mask is built in 
#            this script using the coarse temperature.
#
# path_out: String with the adress where saving the image of T unmixed.
#
# Exemple:
# path_index = '/tmp_user/ldota214h/cgranero/CATUT_unmixing/VNIR-SWIR/4_Index-Estimation/080628/Index_080628_20m.bsq'
# path_temperature = '/tmp_user/ldota214h/cgranero/CATUT_unmixing/TIR/4_TES/Outputs_080628_20m/LST_4_bandes_jour_satellite_20m_bruite.bsq'
# path_fit = '/tmp_user/ldota214h/cgranero/CATUT_unmixing/TsHARP-DISTRAD/Scale_Inv/Fit2_NDVI_T_080628.txt'
# path_out = '/tmp_user/ldota214h/cgranero/CATUT_unmixing/TsHARP-DISTRAD/Scale_Inv/Step2_Images_080628/T_unm_20m_from40m_jourNDVI.bsq'
# path_mask=0
###
######
###########
############### C. Granero-Belinchon; ONERA, Toulouse; 01/2019

	import numpy as np
	from osgeo import gdal, gdalconst

#############################################################################################################################################################################
# Multivariate 4th Order Polynomial regression (HUTS)

	def func(x,p1,p2,p3,p4,p5,p6,p7,p8,p9,p10,p11,p12,p13,p14,p15):
		f = p1*x[0,:,:]**4 + p2*x[0,:,:]**3*x[1,:,:] + p3*x[0,:,:]**2*x[1,:,:]**2 + p4*x[0,:,:]*x[1,:,:]**3 + p5*x[1,:,:]**4 +p6*x[0,:,:]**3 + p7*x[0,:,:]**2*x[1,:,:] + p8*x[0,:,:]*x[1,:,:]**2 + p9*x[1,:,:]**3 + p10*x[0,:,:]**2 + p11*x[0,:,:]*x[1,:,:] + p12*x[1,:,:]**2 + p13*x[0,:,:] + p14*x[1,:,:] + p15
		return f

#############################################################################################################################################################################

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
	
################### ALBEDO ###########################################

# Reading index data
	filename_albedo = path_albedo
	dataset_alb = gdal.Open(filename_albedo)

	cols_alb = dataset_alb.RasterXSize #Spatial dimension x
	rows_alb = dataset_alb.RasterYSize #Spatial dimension y

# Read as array the index
	albedo = dataset_alb.ReadAsArray(0, 0, cols_alb, rows_alb).astype(np.float)

################### Temperature ######################################

# Reading temperature data
	TempFile = path_temperature
	dataset_Temp = gdal.Open(TempFile)

	cols_t = dataset_Temp.RasterXSize #Spatial dimension x
	rows_t = dataset_Temp.RasterYSize #Spatial dimension y

# Read as array
	Temp= dataset_Temp.ReadAsArray(0, 0, cols_t, rows_t).astype(np.float)
	
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
	
	xdata = np.zeros((2,len(index),len(index[0])))
	xdata[0,:,:] = index
	xdata[1,:,:] = albedo
	
	T_unm = func(xdata,Param[0],Param[1],Param[2],Param[3],Param[4],Param[5],Param[6],Param[7],Param[8],Param[9],Param[10],Param[11],Param[12],Param[13],Param[14])

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
	#I_mask = np.greater(Temp,0) # We eliminate the background pixels (not in the image)
	#for icol in range(cols_t):
	#	for irow in range(rows_t):
	#		if I_mask[irow,icol] == False:
	#			T_unm[irow,icol]=0
		
###################################################################
########### SAVING RESULTS ########################################
###################################################################

	driver = dataset_Temp.GetDriver()
	outDs = driver.Create(path_out, int(cols), int(rows), 1, gdal.GDT_Float32)

	outBand = outDs.GetRasterBand(1)
	outBand.WriteArray(T_unm, 0, 0)
	outBand.FlushCache()

	print('Unmixing Done')
	
	return
	
def aatprk_unmixing(path_index, path_temperature, path_slope, path_intercept, iscale, path_out):
	
############## 
#
# This function takes two images (path_index and path_temperature) and the linear regression 
# parameters on the text file (path_fit) and applies $T = a1*I +a0$, where T is the unknown
# a1 and a0 are the regression parameter and I is the index image.
#  	
############## Inputs/Outputs:
#
# path_index: String leading to the index image file. (.bsq, .img, .raw)
#             This file must contain only one band.
#
# path_slope: String leading to the slope image file. (.bsq, .img, .raw)
#             This file must contain only one band.
#
# path_intercept: String leading to the intercept image file. (.bsq, .img, .raw)
#             This file must contain only one band.
#
# path_temperature: String leading to the temperature image file. (.bsq, .img, .raw)	
#                   This file must contain only one band. the background pixels
#                   must be == 0
#
# iscale: Ratio between coarse and fine scale. For coarse =40m and fine =20 iscale =2
#
# path_out: String with the adress where saving the image of T unmixed.
#
# Exemple:
# path_intercept= '/tmp_user/ldota214h/cgranero/CATUT_unmixing/AATPRK/Scale_Inv/Step1_Fits_'+date+'/Intercept_40m_NDVI.bsq' # jour
# path_slope='/tmp_user/ldota214h/cgranero/CATUT_unmixing/AATPRK/Scale_Inv/Step1_Fits_'+date+'/Slope_40m_NDVI.bsq' # jour
# path_index = '/tmp_user/ldota214h/cgranero/CATUT_unmixing/VNIR-SWIR/4_Index-Estimation/080628/Index_080628_20m.bsq'
# path_temperature = '/tmp_user/ldota214h/cgranero/CATUT_unmixing/TIR/4_TES/Outputs_080628_20m/LST_4_bandes_jour_satellite_40m_bruite.bsq'
# path_fit = '/tmp_user/ldota214h/cgranero/CATUT_unmixing/TsHARP-DISTRAD/Scale_Inv/Fit2_NDVI_T_080628.txt'
# path_out = '/tmp_user/ldota214h/cgranero/CATUT_unmixing/TsHARP-DISTRAD/Scale_Inv/Step2_Images_080628/T_unm_20m_from40m_jourNDVI.bsq'
# iscale=2
###
######
###########
############### C. Granero-Belinchon; ONERA, Toulouse; 01/2019

	import numpy as np
	from osgeo import gdal, gdalconst
	
####################################################################
################ LOADING DATA ######################################
####################################################################

################### INDEX ########################################## # fine scale

# Reading index data
	filename_index = path_index
	dataset = gdal.Open(filename_index)

	cols = dataset.RasterXSize #Spatial dimension x
	rows = dataset.RasterYSize #Spatial dimension y

# Read as array the index
	index = dataset.ReadAsArray(0, 0, cols, rows).astype(np.float)

################### Temperature ###################################### #coarse scale

# Reading temperature data
	TempFile = path_temperature
	dataset_Temp = gdal.Open(TempFile)

	cols_t = dataset_Temp.RasterXSize #Spatial dimension x
	rows_t = dataset_Temp.RasterYSize #Spatial dimension y

# Read as array
	Temp= dataset_Temp.ReadAsArray(0, 0, cols_t, rows_t).astype(np.float)

###################################################################
########### UNMIXING ##############################################
###################################################################
	
################### Intercept ##################################### # coarse scale
	
# Reading of data
	intercept_file = path_intercept

#datasets
	intercept_dataset = gdal.Open(intercept_file)

	cols_int = intercept_dataset.RasterXSize #Spatial dimension x
	rows_int = intercept_dataset.RasterYSize #Spatial dimension y

# Read as array
	intercept = intercept_dataset.ReadAsArray(0, 0, cols_int, rows_int).astype(np.float)


################### Slope ##################################### # coarse scale

# Reading of data
	slope_file = path_slope

#datasets
	slope_dataset = gdal.Open(slope_file)

	cols_slope = slope_dataset.RasterXSize #Spatial dimension x
	rows_slope = slope_dataset.RasterYSize #Spatial dimension y

# Read as array
	slope = slope_dataset.ReadAsArray(0, 0, cols_slope, rows_slope).astype(np.float)
	
####
	# This mask uses coarse information. It is better to use fine information	
	#T_unm=np.zeros((rows,cols))
	
	#for ic in range(int(np.floor(cols/iscale))):
		#for ir in range(int(np.floor(rows/iscale))):
			#if Temp[ir,ic]==0:                                       #### MASK
				#for ir_2 in range(ir*iscale,(ir*iscale+iscale)):
					#for ic_2 in range(ic*iscale,(ic*iscale+iscale)):	
						#T_unm[ir_2,ic_2]=0			
			#else:
				#for ir_2 in range(ir*iscale,(ir*iscale+iscale)):
					#for ic_2 in range(ic*iscale,(ic*iscale+iscale)):
						#T_unm[ir_2,ic_2] = intercept[ir,ic] + slope[ir,ic] * index[ir_2,ic_2]

		
	I_mask=np.greater(np.absolute(index),0.0000000000) # We eliminate the background pixels (not in the image)
	T_unm=np.zeros((rows,cols))
	for ic in range(int(np.floor(cols/iscale))):
		for ir in range(int(np.floor(rows/iscale))):
			for ir_2 in range(ir*iscale,(ir*iscale+iscale)):
				for ic_2 in range(ic*iscale,(ic*iscale+iscale)):
					if I_mask[ir_2,ic_2] == False:		
						T_unm[ir_2,ic_2]=0			
					else:
						T_unm[ir_2,ic_2] = intercept[ir,ic] + slope[ir,ic] * index[ir_2,ic_2]
		
###################################################################
########### SAVING RESULTS ######################################## # fine scale
###################################################################

	driver = dataset_Temp.GetDriver()
	outDs = driver.Create(path_out, int(cols), int(rows), 1, gdal.GDT_Float32)

	outBand = outDs.GetRasterBand(1)
	outBand.WriteArray(T_unm, 0, 0)
	outBand.FlushCache()

	print('Unmixing Done')
	
	return

def aatprk_unmixing_v2(path_index, path_temperature, path_slope, path_intercept, iscale,scc, block_size, sill, ran, path_out, path_out_slope, path_out_intercept, path_plot):
    
    ##############
    #
    # This function takes two images (path_index and path_temperature) and the linear regression
    # parameters on the text file (path_fit) and applies $T = a1*I +a0$, where T is the unknown
    # a1 and a0 are the regression parameter and I is the index image.
    #
    ############## Inputs/Outputs:
    #
    # path_index: String leading to the index image file. (.bsq, .img, .raw)
    #             This file must contain only one band.
    #
    # path_slope: String leading to the slope image file. (.bsq, .img, .raw)
    #             This file must contain only one band.
    #
    # path_intercept: String leading to the intercept image file. (.bsq, .img, .raw)
    #             This file must contain only one band.
    #
    # path_temperature: String leading to the temperature image file. (.bsq, .img, .raw)
    #                   This file must contain only one band. the background pixels
    #                   must be == 0
    #
    # iscale: Ratio between coarse and fine scale. For coarse =40m and fine =20 iscale =2
    #
    # scc: The coarse pixel has a size scc * scc
    #
    # block_size: The moving window used to compute the semivariograms and the weights in the
    #             ATPK procedures has a size block_size * block_size
    #
    # sill and ran are the parameters defining the shape of the semivariogram. see line 199
    #
    # path_plot:
    #
    # path_out: String with the adress where saving the image of T unmixed.
    #
    # path_out_slope: String with the adress where saving the image of fine slopes.
    #
    # path_out_intercept: String with the adress where saving the image of fine intercepts.
    #
    # Exemple:
    # path_intercept= '/tmp_user/ldota214h/cgranero/CATUT_unmixing/AATPRK/Scale_Inv/Step1_Fits_'+date+'/Intercept_40m_NDVI.bsq' # jour
    # path_slope='/tmp_user/ldota214h/cgranero/CATUT_unmixing/AATPRK/Scale_Inv/Step1_Fits_'+date+'/Slope_40m_NDVI.bsq' # jour
    # path_index = '/tmp_user/ldota214h/cgranero/CATUT_unmixing/VNIR-SWIR/4_Index-Estimation/080628/Index_080628_20m.bsq'
    # path_temperature = '/tmp_user/ldota214h/cgranero/CATUT_unmixing/TIR/4_TES/Outputs_080628_20m/LST_4_bandes_jour_satellite_40m_bruite.bsq'
    # path_fit = '/tmp_user/ldota214h/cgranero/CATUT_unmixing/TsHARP-DISTRAD/Scale_Inv/Fit2_NDVI_T_080628.txt'
    # path_out = '/tmp_user/ldota214h/cgranero/CATUT_unmixing/TsHARP-DISTRAD/Scale_Inv/Step2_Images_080628/T_unm_20m_from40m_jourNDVI.bsq'
    # iscale=2
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
    
    ################### INDEX ########################################## # fine scale
    
    # Reading index data
    filename_index = path_index
    dataset = gdal.Open(filename_index)
    
    cols = dataset.RasterXSize #Spatial dimension x
    rows = dataset.RasterYSize #Spatial dimension y
    
    # Read as array the index
    index = dataset.ReadAsArray(0, 0, cols, rows).astype(np.float)
    
    ################### Temperature ###################################### #coarse scale
    
    # Reading temperature data
    TempFile = path_temperature
    dataset_Temp = gdal.Open(TempFile)
    
    cols_t = dataset_Temp.RasterXSize #Spatial dimension x
    rows_t = dataset_Temp.RasterYSize #Spatial dimension y
    
    # Read as array
    Temp= dataset_Temp.ReadAsArray(0, 0, cols_t, rows_t).astype(np.float)
    
    ###################################################################
    ########### UNMIXING ##############################################
    ###################################################################
    
    ################### Intercept ##################################### # coarse scale
    
    # Reading of data
    intercept_file = path_intercept
    
    #datasets
    intercept_dataset = gdal.Open(intercept_file)
    
    cols_int = intercept_dataset.RasterXSize #Spatial dimension x
    rows_int = intercept_dataset.RasterYSize #Spatial dimension y
    
    # Read as array
    intercept = intercept_dataset.ReadAsArray(0, 0, cols_int, rows_int).astype(np.float)
    
    
    ################### Slope ##################################### # coarse scale
    
    # Reading of data
    slope_file = path_slope
    
    #datasets
    slope_dataset = gdal.Open(slope_file)
    
    cols_slope = slope_dataset.RasterXSize #Spatial dimension x
    rows_slope = slope_dataset.RasterYSize #Spatial dimension y
    
    # Read as array
    slope = slope_dataset.ReadAsArray(0, 0, cols_slope, rows_slope).astype(np.float)
    
    
    b_radius = int(np.floor(block_size/2))


    #####################################################################
    ################ ATPK METHOD SLOPE ##################################
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
            dt = slope[irow-b_radius:irow+b_radius+1,icol-b_radius:icol+b_radius+1]
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

    mask_II_unm=np.zeros((rows,cols))
    #indice_mask_TT=np.nonzero(TT_unm)
    I_mask=np.greater(np.absolute(index),0.0000000000) # We eliminate the background pixels (not in the image)
    mask_II_unm[I_mask]=1

    # We obtain the delta at the scale of the unmixed temperature
    Slope_final =np.zeros((rows,cols))
        
    for ic in range(b_radius,math.floor(cols/iscale)-b_radius):
        for ir in range(b_radius,math.floor(rows/iscale)-b_radius):
            for ir_2 in range(ir*iscale,(ir*iscale+iscale)):
                for ic_2 in range(ic*iscale,(ic*iscale+iscale)):
                    if mask_II_unm[ir_2,ic_2]==0:
                        Slope_final[ir_2,ic_2] = 0
                    else:
                        temp_var = np.reshape(slope[ir-b_radius:ir+b_radius+1,ic-b_radius:ic+b_radius+1],block_size**2) # We take the block of coarse pixels
                            # We multiply each coarse pixel in the block by the corresponding lambda and sum
                        Slope_final[ir_2,ic_2] = np.sum(lambdas[(ir_2-ir*iscale)*iscale+ic_2-ic*iscale,:]*temp_var)

        
    ###################################################################
    ########### SAVING RESULTS ########################################
    ###################################################################

    driver = dataset.GetDriver()
    outDs = driver.Create(path_out_slope, int(cols), int(rows), 1, gdal.GDT_Float32)
    
    outBand = outDs.GetRasterBand(1)
    outBand.WriteArray(Slope_final, 0, 0)
    outBand.FlushCache()

        
    print('Slope ATPK Done')


    #####################################################################
    ################ ATPK METHOD INTERCEPT ##############################
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
            dt = intercept[irow-b_radius:irow+b_radius+1,icol-b_radius:icol+b_radius+1]
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

    mask_II_unm=np.zeros((rows,cols))
    #indice_mask_TT=np.nonzero(TT_unm)
    I_mask=np.greater(np.absolute(index),0.0000000000) # We eliminate the background pixels (not in the image)
    mask_II_unm[I_mask]=1
    
    # We obtain the delta at the scale of the unmixed temperature
    Intercept_final =np.zeros((rows,cols))
    
    for ic in range(b_radius,math.floor(cols/iscale)-b_radius):
        for ir in range(b_radius,math.floor(rows/iscale)-b_radius):
            for ir_2 in range(ir*iscale,(ir*iscale+iscale)):
                for ic_2 in range(ic*iscale,(ic*iscale+iscale)):
                    if mask_II_unm[ir_2,ic_2]==0:
                        Intercept_final[ir_2,ic_2] = 0
                    else:
                        temp_var = np.reshape(intercept[ir-b_radius:ir+b_radius+1,ic-b_radius:ic+b_radius+1],block_size**2) # We take the block of coarse pixels
                        # We multiply each coarse pixel in the block by the corresponding lambda and sum
                        Intercept_final[ir_2,ic_2] = np.sum(lambdas[(ir_2-ir*iscale)*iscale+ic_2-ic*iscale,:]*temp_var)


###################################################################
########### SAVING RESULTS ########################################
###################################################################

    driver = dataset.GetDriver()
    outDs = driver.Create(path_out_intercept, int(cols), int(rows), 1, gdal.GDT_Float32)

    outBand = outDs.GetRasterBand(1)
    outBand.WriteArray(Intercept_final, 0, 0)
    outBand.FlushCache()
    
    
    print('Intercept ATPK Done')

    I_mask=np.greater(np.absolute(index),0.0000000000) # We eliminate the background pixels (not in the image)
    T_unm=np.zeros((rows,cols))
    for ir_2 in range(rows):
        for ic_2 in range(cols):
            if I_mask[ir_2,ic_2] == False:
                T_unm[ir_2,ic_2]=0
            else:
                T_unm[ir_2,ic_2] = Intercept_final[ir_2,ic_2] + Slope_final[ir_2,ic_2] * index[ir_2,ic_2]

###################################################################
########### SAVING RESULTS ######################################## # fine scale
###################################################################

    driver = dataset_Temp.GetDriver()
    outDs = driver.Create(path_out, int(cols), int(rows), 1, gdal.GDT_Float32)

    outBand = outDs.GetRasterBand(1)
    outBand.WriteArray(T_unm, 0, 0)
    outBand.FlushCache()

    print('Unmixing Done')

    return
