def linear_fit(path_index, path_temperature, min_T, path_fit, plot, path_plot):
	
############## 
#
# This function takes two images (path_index and path_temperature) and 
# computes the linear regression parameters $T = a1*I + a0$. It also 
# applies two masks on the data. The first mask eliminates all the 
# pixels with temperatures below min_T, and the second masks eliminates 
# all the pixels with index == nan.
#  	
############## Inputs/Outputs:
#
# path_index: String leading to the index image file. (.bsq, .img, .raw)
#             This file must contain only one band.
#
# path_temperature: String leading to the index image file. (.bsq, .img, .raw)	
#                   This file must contain only one band.
#
# min_T: Minimum expected temperature in the image. All the temerpature 
#        below this one are considered spurious measures and not taken into account in the 
#        regression.	
#
# path_fit: String with the adress where saving the txt file with the regression parameters.
#
# plot: integer. = 0 if we don't want to obtain a plot, =1 if we want to obtain a plot. 
#
# path_plot: String with the adress where saving the plot of T vs I and the linear regression.
#
# Exemple:
# path_index = '/tmp_user/ldota214h/cgranero/CATUT_unmixing/VNIR-SWIR/4_Index-Estimation/080628/Index_080628_20m.bsq'
# path_temperature = '/tmp_user/ldota214h/cgranero/CATUT_unmixing/TIR/4_TES/Outputs_080628_20m/LST_4_bandes_jour_satellite_20m_bruite.bsq'
# min_T = 285
# path_fit = '/tmp_user/ldota214h/cgranero/CATUT_unmixing/TsHARP-DISTRAD/Scale_Inv/Fit2_NDVI_T_080628.txt'
# plot =1
# path_plot = '/tmp_user/ldota214h/cgranero/CATUT_unmixing/TsHARP-DISTRAD/Scale_Inv/Figs_080628/NDVI_vs_T_080628.png'

################ Needed packages #### 
# import numpy as np
# import matplotlib.pyplot as plt
# from osgeo import gdal, gdalconst
# from scipy.stats import linregress
#
###
######
###########
############### C. Granero-Belinchon; ONERA, Toulouse; 01/2019

	from osgeo import gdal
	import numpy as np
	from scipy.stats import linregress
	import matplotlib.pyplot as plt
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
		
		plt.figure(1)
		plt.scatter(I,T,s=1,c='blue')
		plt.plot(I,fit_fn(I),'--k')
		plt.xlabel('I')
		plt.ylabel('T')
		plt.grid()
		plt.savefig(path_plot)
		plt.close()
		
		print('Fit Done')
		
	else:
		
		print('Fit Done')
	
	return


def bilinear_fit(path_index1, path_index2, path_temperature, min_T, path_fit):
	
############## 
#
# This function takes three images (path_index1, apth_index2 and path_temperature) and 
# computes the linear regression parameters $T = a1*I1 +a2*I2 + a0$. It also 
# applies two masks on the data. The first mask eliminates all the 
# pixels with temperatures below min_T, and the second masks eliminates 
# all the pixels with index1 == nan or index2 ==nan.
#  	
############## Inputs/Outputs:
#
# path_index: String leading to the index image file. (.bsq, .img, .raw)
#             This file must contain only one band.
#
# path_temperature: String leading to the index image file. (.bsq, .img, .raw)	
#                   This file must contain only one band.
#
# min_T: Minimum expected temperature in the image. All the temerpature 
#        below this one are considered spurious measures and not taken into account in the 
#        regression.	
#
# path_fit: String with the adress where saving the txt file with the regression parameters.
#
# Exemple:
# path_index1 = '/tmp_user/ldota214h/cgranero/CATUT_unmixing/VNIR-SWIR/4_Index-Estimation/080628/Index_080628_20m.bsq'
# path_index2 = '/tmp_user/ldota214h/cgranero/CATUT_unmixing/VNIR-SWIR/4_Index-Estimation/080628/Index_080628_20m.bsq'
# path_temperature = '/tmp_user/ldota214h/cgranero/CATUT_unmixing/TIR/4_TES/Outputs_080628_20m/LST_4_bandes_jour_satellite_20m_bruite.bsq'
# min_T = 285
# path_fit = '/tmp_user/ldota214h/cgranero/CATUT_unmixing/TsHARP-DISTRAD/Scale_Inv/Fit2_NDVI_T_080628.txt'
###
######
###########
############### C. Granero-Belinchon; ONERA, Toulouse; 01/2019
	
	from osgeo import gdal, gdalconst
	import numpy as np	
	import scipy.optimize as opt # pour Levenberg-Marquardt
	
################# DEFINITION OF THE BILINEAR EXPRESSION ##############
	
	def func(x,p0,p1,p2):
		f = p1*x[:,0] + p2*x[:,1] + p0
		return f
	
################### INDEX 1 ##########################################
	
# Reading index data
	filename_index1 = path_index1
	dataset1 = gdal.Open(filename_index1)

	cols1 = dataset1.RasterXSize #Spatial dimension x
	rows1 = dataset1.RasterYSize #Spatial dimension y

# Read as array the index
	index1 = dataset1.ReadAsArray(0, 0, cols1, rows1).astype(np.float)
	
################### INDEX 2 ##########################################

# Reading index data
	filename_index2 = path_index2
	dataset2 = gdal.Open(filename_index2)

	cols2 = dataset2.RasterXSize #Spatial dimension x
	rows2 = dataset2.RasterYSize #Spatial dimension y

# Read as array the index
	index2 = dataset2.ReadAsArray(0, 0, cols2, rows2).astype(np.float)

################### Temperature ######################################

# Reading temperature data
	TempFile = path_temperature
	dataset_Temp = gdal.Open(TempFile)

	cols_t = dataset_Temp.RasterXSize #Spatial dimension x
	rows_t = dataset_Temp.RasterYSize #Spatial dimension y

# Read as array
	Temp= dataset_Temp.ReadAsArray(0, 0, cols_t, rows_t).astype(np.float)

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

	param, pcov = opt.curve_fit(func, xdata, ydata,p,method='lm')
	#Result_LM = np.std(ydata-func(xdata,*param))
	
###################################################################
########### SAVING RESULTS ########################################
###################################################################

	np.savetxt(path_fit,(param),fmt='%3.6f')		

###################################################################
########### PLOT ##################################################
###################################################################
		
	print('Fit Done')
	
	return
	
def linear_fit_window(path_index, path_temperature, min_T, b_radius, file_out1, file_out2, path_fit, plot, path_plot):
	
############## 
#
# This function takes two images (path_index and path_temperature) and 
# computes the linear regression parameters $T = a1*I + a0$ within a moving 
# window of size (2*b_radius+1)^2. The obtained slope and intercept values 
# are assigned to the center pixel of the window. It also 
# applies two masks on the data. The first mask eliminates all the 
# pixels with temperatures below min_T, and the second masks eliminates 
# all the pixels with index == nan. The code returns two images, a0 and a1 images, 
# equal in size to the input images.
#  	
############## Inputs/Outputs:
#
# path_index: String leading to the index image file. (.bsq, .img, .raw)
#             This file must contain only one band.
#
# path_temperature: String leading to the index image file. (.bsq, .img, .raw)	
#                   This file must contain only one band.
#
# min_T: Minimum expected temperature in the image. All the temerpature 
#        below this one are considered spurious measures and not taken into account in the 
#        regression.	
#
# b_radius: The moving window used to compute the regression parameters
#            has a size (2*b_radius+1)^2 
#
# path_fit: String with the adress where saving the txt file with the regression parameters.
#
# file_out1: String with the adress where saving the image file with the slope.
#
# file_out2: String with the adress where saving the image file with the slope.
#
# plot: integer. = 0 if we don't want to obtain a plot, =1 if we want to obtain a plot. 
#
# path_plot: String with the adress where saving the plot of T vs I and the linear regression.
#
# Exemple:
# path_index = '/tmp_user/ldota214h/cgranero/CATUT_unmixing/VNIR-SWIR/4_Index-Estimation/080628/Index_080628_20m.bsq'
# path_temperature = '/tmp_user/ldota214h/cgranero/CATUT_unmixing/TIR/4_TES/Outputs_080628_20m/LST_4_bandes_jour_satellite_20m_bruite.bsq'
# min_T = 285
# path_fit = '/tmp_user/ldota214h/cgranero/CATUT_unmixing/TsHARP-DISTRAD/Scale_Inv/Fit2_NDVI_T_080628.txt'
# file_out1='/tmp_user/ldota214h/cgranero/CATUT_unmixing/AATPRK/Scale_Inv/Step1_Fits_080628/Slope_40m_NDVI.bsq'
# file_out2='/tmp_user/ldota214h/cgranero/CATUT_unmixing/AATPRK/Scale_Inv/Step1_Fits_080628/Intercept_40m_NDVI.bsq'
# plot =1
# path_plot = '/tmp_user/ldota214h/cgranero/CATUT_unmixing/TsHARP-DISTRAD/Scale_Inv/Figs_080628/NDVI_vs_T_080628.png'
###
######
###########
############### C. Granero-Belinchon; ONERA, Toulouse; 01/2019

	import numpy as np
	from osgeo import gdal, gdalconst
	from scipy.stats import linregress
	import matplotlib.pyplot as plt

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
	
# Two different linear regressions

	fit2=linregress(I,T)
	fit3 = np.polyfit(I,T,1)
	fit_fn = np.poly1d(fit3)
	
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
			mask=np.greater(Tw,285) # We eliminate the background pixels and those with a temperature under 285K
			Tw=Tw[mask]
			Iw=Iw[mask]
			NaN=np.isfinite(Iw) # We look for nans in the index and we eliminate those pixels from the index and the temperature
			Iw=Iw[NaN]
			Tw=Tw[NaN]
			if len(Tw) > 15:
				fit=linregress(Iw,Tw)
				a1[irow,icol]=fit[0]
				a0[irow,icol]=fit[1]
			if len(Tw) <= 15:
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
			
			
###################################################################
########### SAVING RESULTS ########################################
###################################################################			
				
#1
	driver = dataset_Temp.GetDriver()
	outDs = driver.Create(file_out1, int(cols), int(rows), 1, gdal.GDT_Float32)

	outBand = outDs.GetRasterBand(1)
	outBand.WriteArray(a1, 0, 0)
	outBand.FlushCache()
	
#2
	driver = dataset_Temp.GetDriver()
	outDs = driver.Create(file_out2, int(cols), int(rows), 1, gdal.GDT_Float32)

	outBand = outDs.GetRasterBand(1)
	outBand.WriteArray(a0, 0, 0)
	outBand.FlushCache()	
	
			
	np.savetxt(path_fit,(fit2[0],fit2[1],fit2[2],fit2[3],fit2[4]),fmt='%3.6f')	

###################################################################
########### PLOT ##################################################
###################################################################

	if plot == 1: 
		
		plt.figure(1)
		plt.scatter(I,T,s=1,c='blue')
		plt.plot(I,fit_fn(I),'--k')
		plt.xlabel('I')
		plt.ylabel('T')
		plt.grid()
		plt.savefig(path_plot)
		plt.close()
		
		print('Fit Done')
		
	else:
		
		print('Fit Done')
	
	return
	
def huts_fit(path_index, path_albedo, path_temperature, min_T, path_fit):
	
############## 
#
# This function takes three images (path_index, path_albedo and path_temperature) and 
# computes the polynomial regression parameters of HUTS model. It also 
# applies two masks on the data. The first mask eliminates all the 
# pixels with temperatures below min_T, and the second masks eliminates 
# all the pixels with index1 == nan.
#  	
############## Inputs/Outputs:
#
# path_index: String leading to the index image file. (.bsq, .img, .raw)
#             This file must contain only one band.
#
# path_albedo: String leading to the albedo image file. (.bsq, .img, .raw)
#             This file must contain only one band.
#
# path_temperature: String leading to the index image file. (.bsq, .img, .raw)	
#                   This file must contain only one band.
#
# min_T: Minimum expected temperature in the image. All the temerpature 
#        below this one are considered spurious measures and not taken into account in the 
#        regression.	
#
# path_fit: String with the adress where saving the txt file with the regression parameters.
#
# Exemple:
# path_index1 = '/tmp_user/ldota214h/cgranero/CATUT_unmixing/VNIR-SWIR/4_Index-Estimation/080628/Index_080628_20m.bsq'
# path_albedo = '/tmp_user/ldota214h/cgranero/CATUT_unmixing/HUTS/Albedo_estimation/080628/albedo_080628_20m.bsq'
# path_temperature = '/tmp_user/ldota214h/cgranero/CATUT_unmixing/TIR/4_TES/Outputs_080628_20m/LST_4_bandes_jour_satellite_20m_bruite.bsq'
# min_T = 285
# path_fit = '/tmp_user/ldota214h/cgranero/CATUT_unmixing/TsHARP-DISTRAD/Scale_Inv/Fit2_NDVI_T_080628.txt'
###
######
###########
############### C. Granero-Belinchon; ONERA, Toulouse; 01/2019

	import numpy as np
	import matplotlib.pyplot as plt
	from osgeo import gdal, gdalconst
	import scipy.optimize as opt # pour Levenberg-Marquardt
################# DEFINITION OF THE POLYNOMIAL EXPRESSION ############

# Multivariate 4th Order Polynomial regression (HUTS)

	def func(x,p1,p2,p3,p4,p5,p6,p7,p8,p9,p10,p11,p12,p13,p14,p15):
		f = p1*x[:,0]**4 + p2*x[:,0]**3*x[:,1] + p3*x[:,0]**2*x[:,1]**2 + p4*x[:,0]*x[:,1]**3 + p5*x[:,1]**4 +p6*x[:,0]**3 + p7*x[:,0]**2*x[:,1] + p8*x[:,0]*x[:,1]**2 + p9*x[:,1]**3 + p10*x[:,0]**2 + p11*x[:,0]*x[:,1] + p12*x[:,1]**2 + p13*x[:,0] + p14*x[:,1] + p15
		return f

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

	p = [1.,1.,1.,1.,1.,1.,1.,1.,1.,1.,1.,1.,1.,1.,1.]

	param, pcov = opt.curve_fit(func, xdata, ydata,p,method='lm')
	#Result_LM = np.std(ydata-func(xdata,*param))

	
###################################################################
########### SAVING RESULTS ########################################
###################################################################

	np.savetxt(path_fit,(param),fmt='%3.6f')		

###################################################################
########### PLOT ##################################################
###################################################################
		
	print('Fit Done')
	
	return

def fit_byclass(path_index, path_temperature, path_class, min_T, path_fit, plot, path_plot):
	
############## 
#
# This function takes three images (path_index, path_class and path_temperature) and 
# computes the linear regression parameters $T = a1*I +a0$ separately for each class 
# defined in path_class. It also applies two masks on the data. The first mask 
# eliminates all the pixels with temperatures below min_T, and the second masks eliminates 
# all the pixels with index == nan.
#  	
############## Inputs/Outputs:
#
# path_index: String leading to the index image file. (.bsq, .img, .raw)
#             This file must contain only one band.
#
# path_temperature: String leading to the temperature image file. (.bsq, .img, .raw)	
#                   This file must contain only one band. background pixels =0
#
# path_class : String leading to the classification image file. (.bsq, .img, .raw)
#			   This file must contain only one band. background pixels =0
#
# min_T: Minimum expected temperature in the image. All the temerpature 
#        below this one are considered spurious measures and not taken into account in the 
#        regression.	
#
# path_fit: String with the adress where saving the txt file with the regression parameters.
#
# plot: integer. = 0 if we don't want to obtain a plot, =1 if we want to obtain a plot. 
#
# path_plot: String with the adress where saving the plot of T vs I and the linear regression.
#
# Exemple:
# path_index = '/tmp_user/ldota214h/cgranero/CATUT_unmixing/VNIR-SWIR/4_Index-Estimation/080628/Index_080628_20m.bsq'
# path_temperature = '/tmp_user/ldota214h/cgranero/CATUT_unmixing/TIR/4_TES/Outputs_080628_20m/LST_4_bandes_jour_satellite_20m_bruite.bsq'
# path_class = '/tmp_user/ldota214h/cgranero/CATUT_unmixing/Classif_Unmixing/Index_Based_Classification/Classif_080628/Classif_20m.bsq' 
# min_T = 285
# path_fit = '/tmp_user/ldota214h/cgranero/CATUT_unmixing/TsHARP-DISTRAD/Scale_Inv/Fit2_NDVI_T_080628.txt'
# plot =1
# path_plot = '/tmp_user/ldota214h/cgranero/CATUT_unmixing/Classif_Unmixing/TsHARP_DISTRAD/Scale_Inv/Figs_080628/NDVI_vs_T_080628_20m.png'
###
######
###########
############### C. Granero-Belinchon; ONERA, Toulouse; 01/2019

	import numpy as np
	import matplotlib.pyplot as plt
	from osgeo import gdal, gdalconst
	from scipy.stats import linregress

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

########### FINAL VERSION OF T AND I AND LINEAR REGRESSION ##########

	T = Temp.flatten()
	I = index.flatten() 
	C = Class.flatten()
	
# Classification 
	
	for i in range(num_class):
		
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
		
		fit_c = np.polyfit(Ic,Tc,1)
		fit_fn_c = np.poly1d(fit_c) 
		fit2_c=linregress(Ic,Tc)

		path_text = path_fit[:-4] + "_class"+ str(i+1) + path_fit[-4:]
		np.savetxt(path_text,(fit2_c[0],fit2_c[1],fit2_c[2],fit2_c[3],fit2_c[4]),fmt='%3.6f')

###################################################################
########### PLOT ##################################################
###################################################################

		if plot ==1:

			plt.figure(1)
			plt.scatter(Ic,Tc,s=1,c='blue')
			plt.plot(Ic,fit_fn_c(Ic),'--k')
			plt.xlabel('I')
			plt.ylabel('T')
			plt.grid()
			plt.savefig(path_plot[:-4] + "_class"+ str(i+1) + path_plot[-4:])
			plt.close()
			
			print('Fit by class Done')
		else:
			print('Fit by class Done')
	
	return
	
def linear_fit_ransac(path_index, path_temperature, min_T, iterations, trials, path_fit, plot, path_plot):
	
############## 
#
# This function takes two images (path_index and path_temperature) and 
# computes the RANSAC linear regression parameters $T = a1*I + a0$. It also 
# applies two masks on the data. The first mask eliminates all the 
# pixels with temperatures below min_T, and the second masks eliminates 
# all the pixels with index == nan. The regression doesn't use OLS but 
# Ransac method.
#  	
############## Inputs/Outputs:
#
# path_index: String leading to the index image file. (.bsq, .img, .raw)
#             This file must contain only one band.
#
# path_temperature: String leading to the index image file. (.bsq, .img, .raw)	
#                   This file must contain only one band.
#
# min_T: Minimum expected temperature in the image. All the temerpature 
#        below this one are considered spurious measures and not taken into account in the 
#        regression.	
#
# iterations: Number of iterations to make the slope of ransac method converge in the global regression
#
# trials : Number of iterations inside ransac
#
# path_fit: String with the adress where saving the txt file with the regression parameters.
#
# plot: integer. = 0 if we don't want to obtain a plot, =1 if we want to obtain a plot. 
#
# path_plot: String with the adress where saving the plot of T vs I and the linear regression.
#
# Exemple:
# path_index = '/tmp_user/ldota214h/cgranero/CATUT_unmixing/VNIR-SWIR/4_Index-Estimation/080628/Index_080628_20m.bsq'
# path_temperature = '/tmp_user/ldota214h/cgranero/CATUT_unmixing/TIR/4_TES/Outputs_080628_20m/LST_4_bandes_jour_satellite_20m_bruite.bsq'
# min_T = 285
# iterations=25
# trials=250 (In any case always higher than 100)
# path_fit = '/tmp_user/ldota214h/cgranero/CATUT_unmixing/TsHARP-DISTRAD/Scale_Inv/Fit2_NDVI_T_080628.txt'
# plot =1
# path_plot = '/tmp_user/ldota214h/cgranero/CATUT_unmixing/TsHARP-DISTRAD/Scale_Inv/Figs_080628/NDVI_vs_T_080628.png'
#
# Reference:
# [1] Martin A. Fischler & Robert C. Bolles (June 1981). "Random Sample Consensus: A Paradigm for Model Fitting with Applications to 
#     Image Analysis and Automated Cartography" (PDF). Comm. ACM. 24 (6): 381–395. doi:10.1145/358669.358692.
#
################ Needed packages #### 
# import numpy as np
# import matplotlib.pyplot as plt
# from osgeo import gdal, gdalconst
# from scipy.stats import linregress
# from sklearn import linear_model
#
###
######
###########
############### C. Granero-Belinchon; ONERA, Toulouse; 06/2019

	from osgeo import gdal
	import numpy as np
	from scipy.stats import linregress
	import matplotlib.pyplot as plt
	from sklearn import linear_model
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
	
	# To use Ransac regressor I should be a matrix [n_samples, n_features], and T a vector [n_samples]
	I=I.reshape(-1,1)
# Two different linear regressions
	ransac = linear_model.RANSACRegressor(max_trials=trials)
	intercept=0
	slope=0
	for i in range(iterations):
		ransac.fit(I,T)
		intercept=intercept+ransac.estimator_.intercept_
		slope=slope+ransac.estimator_.coef_[0]
	
	
###################################################################
########### SAVING RESULTS ########################################
###################################################################
				
	np.savetxt(path_fit,(slope/iterations,intercept/iterations),fmt='%3.6f')	

###################################################################
########### PLOT ##################################################
###################################################################

	if plot == 1: 
		
		plt.figure(1)
		plt.scatter(I,T,s=1,c='blue')
		plt.plot(I, intercept/iterations+(slope/iterations)*I, 'k--', linewidth=3)
		plt.xlabel('I')
		plt.ylabel('T')
		plt.grid()
		plt.savefig(path_plot)
		plt.close()
		
		print('Fit Done')
		
	else:
		
		print('Fit Done')
	
	return	
	
def linear_fit_window_ransac(path_index, path_temperature, min_T, b_radius, iterations, trials, file_out1, file_out2, path_fit):
	
############## 
#
# This function takes two images (path_index and path_temperature) and 
# computes the Ransac linear regression parameters $T = a1*I + a0$ within a moving 
# window of size (2*b_radius+1)^2. The obtained slope and intercept values 
# are assigned to the center pixel of the window. It also 
# applies two masks on the data. The first mask eliminates all the 
# pixels with temperatures below min_T, and the second masks eliminates 
# all the pixels with index == nan. The code returns two images, a0 and a1 images, 
# equal in size to the input images.
#  	
############## Inputs/Outputs:
#
# path_index: String leading to the index image file. (.bsq, .img, .raw)
#             This file must contain only one band.
#
# path_temperature: String leading to the index image file. (.bsq, .img, .raw)	
#                   This file must contain only one band.
#
# min_T: Minimum expected temperature in the image. All the temerpature 
#        below this one are considered spurious measures and not taken into account in the 
#        regression.	
#
# b_radius: The moving window used to compute the regression parameters
#            has a size (2*b_radius+1)^2 
#
# iterations: Number of iterations to make the slope of ransac method converge in the global regression
#
# trials : Number of iterations inside ransac
#
# path_fit: String with the adress where saving the txt file with the regression parameters.
#
# file_out1: String with the adress where saving the image file with the slope.
#
# file_out2: String with the adress where saving the image file with the slope.
#
#
# Exemple:
# path_index = '/tmp_user/ldota214h/cgranero/CATUT_unmixing/VNIR-SWIR/4_Index-Estimation/080628/Index_080628_20m.bsq'
# path_temperature = '/tmp_user/ldota214h/cgranero/CATUT_unmixing/TIR/4_TES/Outputs_080628_20m/LST_4_bandes_jour_satellite_20m_bruite.bsq'
# min_T = 285
# trials=250 (In any case always higher than 100)
# iterations=25
# path_fit = '/tmp_user/ldota214h/cgranero/CATUT_unmixing/TsHARP-DISTRAD/Scale_Inv/Fit2_NDVI_T_080628.txt'
# file_out1='/tmp_user/ldota214h/cgranero/CATUT_unmixing/AATPRK/Scale_Inv/Step1_Fits_080628/Slope_40m_NDVI.bsq'
# file_out2='/tmp_user/ldota214h/cgranero/CATUT_unmixing/AATPRK/Scale_Inv/Step1_Fits_080628/Intercept_40m_NDVI.bsq'
#
# Reference:
# [1] Martin A. Fischler & Robert C. Bolles (June 1981). "Random Sample Consensus: A Paradigm for Model Fitting with Applications to 
#     Image Analysis and Automated Cartography" (PDF). Comm. ACM. 24 (6): 381–395. doi:10.1145/358669.358692.
#
###
######
###########
############### C. Granero-Belinchon; ONERA, Toulouse; 01/2019

	import numpy as np
	from osgeo import gdal, gdalconst
	from scipy.stats import linregress
	import matplotlib.pyplot as plt
	from sklearn import linear_model
	
# "sys.stdout" lines prevent lts_linefit from printing in the console  
	import sys
	import os
	
# Ransac regressor	
	ransac = linear_model.RANSACRegressor(max_trials=trials)

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
	
# Linear regression

	# To use Ransac regressor I should be a matrix [n_samples, n_features], and T a vector [n_samples]
	I=I.reshape(-1,1)
	int_glob=0
	slope_glob=0
	for i in range(iterations):
		ransac.fit(I,T)
		int_glob=int_glob+ransac.estimator_.intercept_
		slope_glob=slope_glob+ransac.estimator_.coef_[0]
	
	global_fit=np.zeros(2)
	global_fit[0]=int_glob/iterations
	global_fit[1]=slope_glob/iterations
	
	local_fit=np.zeros(2)
#####################################################################
################### WINDOW DEFINITION , #############################
########### AND LINEAR REGRESSION FOR EACH WINDOW ###################

	a1 = np.zeros((rows, cols))
	a0 = np.zeros((rows, cols))

	for irow in range(b_radius,rows-b_radius):
		print("irow=",irow)
		print("icol=",end =" ")
		for icol in range(b_radius,cols-b_radius):
			print(icol,end =" ")
			Tw = Temp[irow-b_radius:irow+b_radius+1,icol-b_radius:icol+b_radius+1]
			Iw = index[irow-b_radius:irow+b_radius+1,icol-b_radius:icol+b_radius+1]
			Tw = Tw.flatten()
			Iw = Iw.flatten()
			mask=np.greater(Tw,min_T) # We eliminate the background pixels and those with a temperature under 285K
			Tw=Tw[mask]
			Iw=Iw[mask]
			NaN=np.isfinite(Iw) # We look for nans in the index and we eliminate those pixels from the index and the temperature
			Iw=Iw[NaN]
			Tw=Tw[NaN]
			
			if len(Tw) > 15:
				sys.stdout = open(os.devnull, "w")
				Iw=Iw.reshape(-1,1)
				ransac.fit(Iw,Tw)
				sys.stdout = sys.__stdout__
	
				local_fit[0]=ransac.estimator_.intercept_
				local_fit[1]=ransac.estimator_.coef_[0]
				a1[irow,icol]=local_fit[1]
				a0[irow,icol]=local_fit[0]
			if len(Tw) <= 15:
				a1[irow,icol]=global_fit[1]
				a0[irow,icol]=global_fit[0]
		print(" ")
	
	for irow in range(0,b_radius):
		for icol in range(cols):			
			a1[irow,icol]=global_fit[1]
			a0[irow,icol]=global_fit[0]
			
	for irow in range(rows-b_radius,rows):
		for icol in range(cols):			
			a1[irow,icol]=global_fit[1]
			a0[irow,icol]=global_fit[0]
			
	for icol in range(0,b_radius):
		for irow in range(rows):			
			a1[irow,icol]=global_fit[1]
			a0[irow,icol]=global_fit[0]
			
	for icol in range(cols-b_radius,cols):
		for irow in range(rows):			
			a1[irow,icol]=global_fit[1]
			a0[irow,icol]=global_fit[0]
			
			
###################################################################
########### SAVING RESULTS ########################################
###################################################################			
				
#1
	driver = dataset_Temp.GetDriver()
	outDs = driver.Create(file_out1, int(cols), int(rows), 1, gdal.GDT_Float32)

	outBand = outDs.GetRasterBand(1)
	outBand.WriteArray(a1, 0, 0)
	outBand.FlushCache()
	
#2
	driver = dataset_Temp.GetDriver()
	outDs = driver.Create(file_out2, int(cols), int(rows), 1, gdal.GDT_Float32)

	outBand = outDs.GetRasterBand(1)
	outBand.WriteArray(a0, 0, 0)
	outBand.FlushCache()	
	
			
	np.savetxt(path_fit,(global_fit[1],global_fit[0]),fmt='%3.6f')	
		
	print('Fit Done')
	
	return		
	
def linear_fit_siegel(path_index, path_temperature, min_T, path_fit, plot, path_plot):
	
	###
	   ### MEMORY ERROR (Too big image and small computer)
	###
	
############## 
#
# This function takes two images (path_index and path_temperature) and 
# computes the Siegel linear regression parameters $T = a1*I + a0$. It also 
# applies two masks on the data. The first mask eliminates all the 
# pixels with temperatures below min_T, and the second masks eliminates 
# all the pixels with index == nan. The regression doesn't use OLS but 
# Theilsen method.
#  	
############## Inputs/Outputs:
#
# path_index: String leading to the index image file. (.bsq, .img, .raw)
#             This file must contain only one band.
#
# path_temperature: String leading to the index image file. (.bsq, .img, .raw)	
#                   This file must contain only one band.
#
# min_T: Minimum expected temperature in the image. All the temerpature 
#        below this one are considered spurious measures and not taken into account in the 
#        regression.	
#
# path_fit: String with the adress where saving the txt file with the regression parameters.
#
# plot: integer. = 0 if we don't want to obtain a plot, =1 if we want to obtain a plot. 
#
# path_plot: String with the adress where saving the plot of T vs I and the linear regression.
#
# Exemple:
# path_index = '/tmp_user/ldota214h/cgranero/CATUT_unmixing/VNIR-SWIR/4_Index-Estimation/080628/Index_080628_20m.bsq'
# path_temperature = '/tmp_user/ldota214h/cgranero/CATUT_unmixing/TIR/4_TES/Outputs_080628_20m/LST_4_bandes_jour_satellite_20m_bruite.bsq'
# min_T = 285
# path_fit = '/tmp_user/ldota214h/cgranero/CATUT_unmixing/TsHARP-DISTRAD/Scale_Inv/Fit2_NDVI_T_080628.txt'
# plot =1
# path_plot = '/tmp_user/ldota214h/cgranero/CATUT_unmixing/TsHARP-DISTRAD/Scale_Inv/Figs_080628/NDVI_vs_T_080628.png'
#
# References: 
# [1] A. Siegel, “Robust Regression Using Repeated Medians”, Biometrika, Vol. 69, pp. 242-244, 1982. 
# [2] A. Stein and M. Werman, “Finding the repeated median regression line”, Proceedings of the Third Annual ACM-SIAM Symposium on Discrete Algorithms, pp. 409-413, 1992. 
#
################ Needed packages #### 
# import numpy as np
# import matplotlib.pyplot as plt
# from osgeo import gdal, gdalconst
# from scipy.stats import linregress
# from scipy.stats import siegelslopes
#
###
######
###########
############### C. Granero-Belinchon; ONERA, Toulouse; 06/2019

	from osgeo import gdal
	import numpy as np
	from scipy.stats import linregress
	import matplotlib.pyplot as plt
	from scipy.stats import siegelslopes
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
	
# Two different linear regressions
	sieg = siegelslopes(T,I)
	
###################################################################
########### SAVING RESULTS ########################################
###################################################################
				
	np.savetxt(path_fit,(sieg[0],sieg[1]),fmt='%3.6f')	

###################################################################
########### PLOT ##################################################
###################################################################

	if plot == 1: 
		
		plt.figure(1)
		plt.scatter(I,T,s=1,c='blue')
		plt.plot(I, sieg[1]+sieg[0]*I, 'k--', linewidth=3)
		plt.xlabel('I')
		plt.ylabel('T')
		plt.grid()
		plt.savefig(path_plot)
		plt.close()
		
		print('Fit Done')
		
	else:
		
		print('Fit Done')
	
	return		
	
def linear_fit_window_siegel(path_index, path_temperature, min_T, b_radius, file_out1, file_out2, path_fit):
	
############## 
#
# This function takes two images (path_index and path_temperature) and 
# computes the Siegel linear regression parameters $T = a1*I + a0$ within a moving 
# window of size (2*b_radius+1)^2. The obtained slope and intercept values 
# are assigned to the center pixel of the window. It also 
# applies two masks on the data. The first mask eliminates all the 
# pixels with temperatures below min_T, and the second masks eliminates 
# all the pixels with index == nan. The code returns two images, a0 and a1 images, 
# equal in size to the input images.
#
# Note: In order to avoid Memory Errors, OLS regression is used when Global regression.
#  	
############## Inputs/Outputs:
#
# path_index: String leading to the index image file. (.bsq, .img, .raw)
#             This file must contain only one band.
#
# path_temperature: String leading to the index image file. (.bsq, .img, .raw)	
#                   This file must contain only one band.
#
# min_T: Minimum expected temperature in the image. All the temerpature 
#        below this one are considered spurious measures and not taken into account in the 
#        regression.	
#
# b_radius: The moving window used to compute the regression parameters
#            has a size (2*b_radius+1)^2 
#
# path_fit: String with the adress where saving the txt file with the regression parameters.
#
# file_out1: String with the adress where saving the image file with the slope.
#
# file_out2: String with the adress where saving the image file with the slope.
#
# Exemple:
# path_index = '/tmp_user/ldota214h/cgranero/CATUT_unmixing/VNIR-SWIR/4_Index-Estimation/080628/Index_080628_20m.bsq'
# path_temperature = '/tmp_user/ldota214h/cgranero/CATUT_unmixing/TIR/4_TES/Outputs_080628_20m/LST_4_bandes_jour_satellite_20m_bruite.bsq'
# min_T = 285
# path_fit = '/tmp_user/ldota214h/cgranero/CATUT_unmixing/TsHARP-DISTRAD/Scale_Inv/Fit2_NDVI_T_080628.txt'
# file_out1='/tmp_user/ldota214h/cgranero/CATUT_unmixing/AATPRK/Scale_Inv/Step1_Fits_080628/Slope_40m_NDVI.bsq'
# file_out2='/tmp_user/ldota214h/cgranero/CATUT_unmixing/AATPRK/Scale_Inv/Step1_Fits_080628/Intercept_40m_NDVI.bsq'
#
# References: 
# [1] A. Siegel, “Robust Regression Using Repeated Medians”, Biometrika, Vol. 69, pp. 242-244, 1982. 
# [2] A. Stein and M. Werman, “Finding the repeated median regression line”, Proceedings of the Third Annual ACM-SIAM Symposium on Discrete Algorithms, pp. 409-413, 1992.
###
######
###########
############### C. Granero-Belinchon; ONERA, Toulouse; 01/2019

	import numpy as np
	from osgeo import gdal, gdalconst
	from scipy.stats import linregress
	import matplotlib.pyplot as plt
	from sklearn import linear_model
	from scipy.stats import siegelslopes
# "sys.stdout" lines prevent lts_linefit from printing in the console  
	import sys
	import os

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
	
# Linear regression

	fit2=linregress(I,T)

	global_fit=np.zeros(2)
	global_fit[0]=fit2[1]
	global_fit[1]=fit2[0]
	
	local_fit=np.zeros(2)
#####################################################################
################### WINDOW DEFINITION , #############################
########### AND LINEAR REGRESSION FOR EACH WINDOW ###################

	a1 = np.zeros((rows, cols))
	a0 = np.zeros((rows, cols))

	for irow in range(b_radius,rows-b_radius):
		print("irow=",irow)
		print("icol=",end =" ")
		for icol in range(b_radius,cols-b_radius):
			print(icol,end =" ")
			Tw = Temp[irow-b_radius:irow+b_radius+1,icol-b_radius:icol+b_radius+1]
			Iw = index[irow-b_radius:irow+b_radius+1,icol-b_radius:icol+b_radius+1]
			Tw = Tw.flatten()
			Iw = Iw.flatten()
			mask=np.greater(Tw,min_T) # We eliminate the background pixels and those with a temperature under 285K
			Tw=Tw[mask]
			Iw=Iw[mask]
			NaN=np.isfinite(Iw) # We look for nans in the index and we eliminate those pixels from the index and the temperature
			Iw=Iw[NaN]
			Tw=Tw[NaN]
			
			if len(Tw) > 15:
				sys.stdout = open(os.devnull, "w")
				fit=siegelslopes(Tw,Iw)
				sys.stdout = sys.__stdout__
	
				local_fit[0]=fit[1]
				local_fit[1]=fit[0]
				a1[irow,icol]=local_fit[1]
				a0[irow,icol]=local_fit[0]
			if len(Tw) <= 15:
				a1[irow,icol]=global_fit[1]
				a0[irow,icol]=global_fit[0]
		print(" ")
	
	for irow in range(0,b_radius):
		for icol in range(cols):			
			a1[irow,icol]=global_fit[1]
			a0[irow,icol]=global_fit[0]
			
	for irow in range(rows-b_radius,rows):
		for icol in range(cols):			
			a1[irow,icol]=global_fit[1]
			a0[irow,icol]=global_fit[0]
			
	for icol in range(0,b_radius):
		for irow in range(rows):			
			a1[irow,icol]=global_fit[1]
			a0[irow,icol]=global_fit[0]
			
	for icol in range(cols-b_radius,cols):
		for irow in range(rows):			
			a1[irow,icol]=global_fit[1]
			a0[irow,icol]=global_fit[0]
			
			
###################################################################
########### SAVING RESULTS ########################################
###################################################################			
				
#1
	driver = dataset_Temp.GetDriver()
	outDs = driver.Create(file_out1, int(cols), int(rows), 1, gdal.GDT_Float32)

	outBand = outDs.GetRasterBand(1)
	outBand.WriteArray(a1, 0, 0)
	outBand.FlushCache()
	
#2
	driver = dataset_Temp.GetDriver()
	outDs = driver.Create(file_out2, int(cols), int(rows), 1, gdal.GDT_Float32)

	outBand = outDs.GetRasterBand(1)
	outBand.WriteArray(a0, 0, 0)
	outBand.FlushCache()	
	
			
	#np.savetxt(path_fit,(p.ab[1],p.ab[0]),fmt='%3.6f')	
		
	print('Fit Done')
	
	return	
	
def linear_fit_theilsen(path_index, path_temperature, min_T, path_fit, plot, path_plot):
	
	###
	   ### MEMORY ERROR (Too big image and small computer)
	###
	
	
############## 
#
# This function takes two images (path_index and path_temperature) and 
# computes the Theil Sen linear regression parameters $T = a1*I + a0$. It also 
# applies two masks on the data. The first mask eliminates all the 
# pixels with temperatures below min_T, and the second masks eliminates 
# all the pixels with index == nan. The regression doesn't use OLS but 
# Theilsen method.
#  	
############## Inputs/Outputs:
#
# path_index: String leading to the index image file. (.bsq, .img, .raw)
#             This file must contain only one band.
#
# path_temperature: String leading to the index image file. (.bsq, .img, .raw)	
#                   This file must contain only one band.
#
# min_T: Minimum expected temperature in the image. All the temerpature 
#        below this one are considered spurious measures and not taken into account in the 
#        regression.	
#
# path_fit: String with the adress where saving the txt file with the regression parameters.
#
# plot: integer. = 0 if we don't want to obtain a plot, =1 if we want to obtain a plot. 
#
# path_plot: String with the adress where saving the plot of T vs I and the linear regression.
#
# Exemple:
# path_index = '/tmp_user/ldota214h/cgranero/CATUT_unmixing/VNIR-SWIR/4_Index-Estimation/080628/Index_080628_20m.bsq'
# path_temperature = '/tmp_user/ldota214h/cgranero/CATUT_unmixing/TIR/4_TES/Outputs_080628_20m/LST_4_bandes_jour_satellite_20m_bruite.bsq'
# min_T = 285
# path_fit = '/tmp_user/ldota214h/cgranero/CATUT_unmixing/TsHARP-DISTRAD/Scale_Inv/Fit2_NDVI_T_080628.txt'
# plot =1
# path_plot = '/tmp_user/ldota214h/cgranero/CATUT_unmixing/TsHARP-DISTRAD/Scale_Inv/Figs_080628/NDVI_vs_T_080628.png'
#
# References: 
# [1] Theil-Sen Estimators in a Multiple Linear Regression Model, 2009 Xin Dang, Hanxiang Peng, Xueqin Wang and Heping Zhang
#
################ Needed packages #### 
# import numpy as np
# import matplotlib.pyplot as plt
# from osgeo import gdal, gdalconst
# from scipy.stats import linregress
# from scipy.stats import siegelslopes
#
###
######
###########
############### C. Granero-Belinchon; ONERA, Toulouse; 06/2019

	from osgeo import gdal
	import numpy as np
	from scipy.stats import linregress
	import matplotlib.pyplot as plt
	from scipy.stats import theilslopes
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
	
# Two different linear regressions
	theil = theilslopes(T,I)
	
###################################################################
########### SAVING RESULTS ########################################
###################################################################
				
	np.savetxt(path_fit,(theil[0],theil[1]),fmt='%3.6f')	

###################################################################
########### PLOT ##################################################
###################################################################

	if plot == 1: 
		
		plt.figure(1)
		plt.scatter(I,T,s=1,c='blue')
		plt.plot(I, theil[1]+theil[0]*I, 'k--', linewidth=3)
		plt.xlabel('I')
		plt.ylabel('T')
		plt.grid()
		plt.savefig(path_plot)
		plt.close()
		
		print('Fit Done')
		
	else:
		
		print('Fit Done')
	
	return		
	
def linear_fit_window_theilsen(path_index, path_temperature, min_T, b_radius, file_out1, file_out2, path_fit):
	
############## 
#
# This function takes two images (path_index and path_temperature) and 
# computes the Theil Sen linear regression parameters $T = a1*I + a0$ within a moving 
# window of size (2*b_radius+1)^2. The obtained slope and intercept values 
# are assigned to the center pixel of the window. It also 
# applies two masks on the data. The first mask eliminates all the 
# pixels with temperatures below min_T, and the second masks eliminates 
# all the pixels with index == nan. The code returns two images, a0 and a1 images, 
# equal in size to the input images.
#
# Note: In order to avoid Memory Errors, OLS regression is used when Global regression.
#  	
############## Inputs/Outputs:
#
# path_index: String leading to the index image file. (.bsq, .img, .raw)
#             This file must contain only one band.
#
# path_temperature: String leading to the index image file. (.bsq, .img, .raw)	
#                   This file must contain only one band.
#
# min_T: Minimum expected temperature in the image. All the temerpature 
#        below this one are considered spurious measures and not taken into account in the 
#        regression.	
#
# b_radius: The moving window used to compute the regression parameters
#            has a size (2*b_radius+1)^2 
#
# path_fit: String with the adress where saving the txt file with the regression parameters.
#
# file_out1: String with the adress where saving the image file with the slope.
#
# file_out2: String with the adress where saving the image file with the slope.
#
# Exemple:
# path_index = '/tmp_user/ldota214h/cgranero/CATUT_unmixing/VNIR-SWIR/4_Index-Estimation/080628/Index_080628_20m.bsq'
# path_temperature = '/tmp_user/ldota214h/cgranero/CATUT_unmixing/TIR/4_TES/Outputs_080628_20m/LST_4_bandes_jour_satellite_20m_bruite.bsq'
# min_T = 285
# path_fit = '/tmp_user/ldota214h/cgranero/CATUT_unmixing/TsHARP-DISTRAD/Scale_Inv/Fit2_NDVI_T_080628.txt'
# file_out1='/tmp_user/ldota214h/cgranero/CATUT_unmixing/AATPRK/Scale_Inv/Step1_Fits_080628/Slope_40m_NDVI.bsq'
# file_out2='/tmp_user/ldota214h/cgranero/CATUT_unmixing/AATPRK/Scale_Inv/Step1_Fits_080628/Intercept_40m_NDVI.bsq'
#
# References: 
# [1] Theil-Sen Estimators in a Multiple Linear Regression Model, 2009 Xin Dang, Hanxiang Peng, Xueqin Wang and Heping Zhang
# 
###
######
###########
############### C. Granero-Belinchon; ONERA, Toulouse; 01/2019

	import numpy as np
	from osgeo import gdal, gdalconst
	from scipy.stats import linregress
	import matplotlib.pyplot as plt
	from sklearn import linear_model
	from scipy.stats import theilslopes
# "sys.stdout" lines prevent lts_linefit from printing in the console  
	import sys
	import os

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
	
# Linear regression

	fit2=linregress(I,T)

	global_fit=np.zeros(2)
	global_fit[0]=fit2[1]
	global_fit[1]=fit2[0]
	
	local_fit=np.zeros(2)
#####################################################################
################### WINDOW DEFINITION , #############################
########### AND LINEAR REGRESSION FOR EACH WINDOW ###################

	a1 = np.zeros((rows, cols))
	a0 = np.zeros((rows, cols))

	for irow in range(b_radius,rows-b_radius):
		print("irow=",irow)
		print("icol=",end =" ")
		for icol in range(b_radius,cols-b_radius):
			print(icol,end =" ")
			Tw = Temp[irow-b_radius:irow+b_radius+1,icol-b_radius:icol+b_radius+1]
			Iw = index[irow-b_radius:irow+b_radius+1,icol-b_radius:icol+b_radius+1]
			Tw = Tw.flatten()
			Iw = Iw.flatten()
			mask=np.greater(Tw,min_T) # We eliminate the background pixels and those with a temperature under 285K
			Tw=Tw[mask]
			Iw=Iw[mask]
			NaN=np.isfinite(Iw) # We look for nans in the index and we eliminate those pixels from the index and the temperature
			Iw=Iw[NaN]
			Tw=Tw[NaN]
			
			if len(Tw) > 15:
				sys.stdout = open(os.devnull, "w")
				fit=theilslopes(Tw,Iw)
				sys.stdout = sys.__stdout__
	
				local_fit[0]=fit[1]
				local_fit[1]=fit[0]
				a1[irow,icol]=local_fit[1]
				a0[irow,icol]=local_fit[0]
			if len(Tw) <= 15:
				a1[irow,icol]=global_fit[1]
				a0[irow,icol]=global_fit[0]
		print(" ")
	
	for irow in range(0,b_radius):
		for icol in range(cols):			
			a1[irow,icol]=global_fit[1]
			a0[irow,icol]=global_fit[0]
			
	for irow in range(rows-b_radius,rows):
		for icol in range(cols):			
			a1[irow,icol]=global_fit[1]
			a0[irow,icol]=global_fit[0]
			
	for icol in range(0,b_radius):
		for irow in range(rows):			
			a1[irow,icol]=global_fit[1]
			a0[irow,icol]=global_fit[0]
			
	for icol in range(cols-b_radius,cols):
		for irow in range(rows):			
			a1[irow,icol]=global_fit[1]
			a0[irow,icol]=global_fit[0]
			
			
###################################################################
########### SAVING RESULTS ########################################
###################################################################			
				
#1
	driver = dataset_Temp.GetDriver()
	outDs = driver.Create(file_out1, int(cols), int(rows), 1, gdal.GDT_Float32)

	outBand = outDs.GetRasterBand(1)
	outBand.WriteArray(a1, 0, 0)
	outBand.FlushCache()
	
#2
	driver = dataset_Temp.GetDriver()
	outDs = driver.Create(file_out2, int(cols), int(rows), 1, gdal.GDT_Float32)

	outBand = outDs.GetRasterBand(1)
	outBand.WriteArray(a0, 0, 0)
	outBand.FlushCache()	
	
			
	#np.savetxt(path_fit,(p.ab[1],p.ab[0]),fmt='%3.6f')	
		
	print('Fit Done')
	
	return	
	
def linear_fit_lts(path_index, path_temperature, min_T, sigI, sigT, path_fit, plot, path_plot):
	
############## 
#
# This function takes two images (path_index and path_temperature) and 
# computes the linear regression parameters $T = a1*I + a0$. It also 
# applies two masks on the data. The first mask eliminates all the 
# pixels with temperatures below min_T, and the second masks eliminates 
# all the pixels with index == nan. The linear regression is not the 
# Ordinary Least Squares but the LTS of Capellari 2013 [1]. This kind of 
# regression doesn't take into account the outliers.
#
#  	
############## Inputs/Outputs:
#
# path_index: String leading to the index image file. (.bsq, .img, .raw)
#             This file must contain only one band.
#
# path_temperature: String leading to the index image file. (.bsq, .img, .raw)	
#                   This file must contain only one band.
#
# min_T: Minimum expected temperature in the image. All the temerpature 
#        below this one are considered spurious measures and not taken into account in the 
#        regression.	
#
# sigI : Error in the dimension x of the measures (Index)
#
# sigT : Error in the dimension y of the measures (Temperature)
#
# pivot : pivot = np.median(x) see [1]
#
# path_fit: String with the adress where saving the txt file with the regression parameters.
#
# plot: integer. = 0 if we don't want to obtain a plot, =1 if we want to obtain a plot. 
#
# path_plot: String with the adress where saving the plot of T vs I and the linear regression.
#
# Exemple:
# path_index = '/tmp_user/ldota214h/cgranero/CATUT_unmixing/TEST_Mod/Index/NDBI_080628.img'
# path_temperature = '/tmp_user/ldota214h/cgranero/CATUT_unmixing/TIR/4_TES/Outputs_080628_40m/LST_4_bandes_jour_satellite_40m_bruite.bsq'
# min_T = 285
# sigx =0.05 
# sigy = 1
# #pivot= np.median(x)
# path_fit = '/tmp_user/ldota214h/cgranero/CATUT_unmixing/TsHARP-DISTRAD/Scale_Inv/Fit2_NDVI_T_080628.txt'
# plot =1
# path_plot = '/tmp_user/ldota214h/cgranero/CATUT_unmixing/TsHARP-DISTRAD/Scale_Inv/Figs_080628/NDVI_vs_T_080628.png'
#
# Reference:
# [1] Capellari et al. The ATLAS$^{3D}$ project - XV. Benchmark for early-type galaxies scaling relations from 260 dynamical models: 
# mass-to-light ratio, dark matter, Fundamental Plane and Mass Plane. MNRAS (2013) 
#
################ Needed packages #### 
# import numpy as np
# import matplotlib.pyplot as plt
# from osgeo import gdal, gdalconst
# from scipy.stats import linregress
# from ltsfit.lts_linefit import lts_linefit
#
###
######
###########
############### C. Granero-Belinchon; ONERA, Toulouse; 06/2019

	from osgeo import gdal
	import numpy as np
	import matplotlib.pyplot as plt
	from ltsfit.lts_linefit import lts_linefit
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
	
# Two different linear regressions
	sigI=np.full_like(I, sigI)
	sigT=np.full_like(I, sigT)

	p = lts_linefit(I, T, sigI, sigT, pivot=np.median(I))
	
###################################################################
########### SAVING RESULTS ########################################
###################################################################
				
	np.savetxt(path_fit,(p.ab[1],p.ab[0]),fmt='%3.6f')	

###################################################################
########### PLOT ##################################################
###################################################################

	if plot == 1: 
		
		plt.xlabel('x')
		plt.ylabel('$y, a + b\ (x - x_{0})$')
		plt.pause(1)
		plt.savefig(path_plot)
		plt.close()
		
		print('Fit Done')
		
	else:
		
		print('Fit Done')
	
	return	
	
def linear_fit_window_lts(path_index, path_temperature, min_T, b_radius, file_out1, file_out2, path_fit, sigI,sigT, min_slope, max_slope, min_intercept, max_intercept):
	
	
############## 
#
# This function takes two images (path_index and path_temperature) and 
# computes the lts linear regression parameters $T = a1*I + a0$ within a moving 
# window of size (2*b_radius+1)^2. The obtained slope and intercept values 
# are assigned to the center pixel of the window. It also 
# applies two masks on the data. The first mask eliminates all the 
# pixels with temperatures below min_T, and the second masks eliminates 
# all the pixels with index == nan. The code returns two images, a0 and a1 images, 
# equal in size to the input images.
#  	
############## Inputs/Outputs:
#
# path_index: String leading to the index image file. (.bsq, .img, .raw)
#             This file must contain only one band.
#
# path_temperature: String leading to the index image file. (.bsq, .img, .raw)	
#                   This file must contain only one band.
#
# min_T: Minimum expected temperature in the image. All the temerpature 
#        below this one are considered spurious measures and not taken into account in the 
#        regression.	
#
# b_radius: The moving window used to compute the regression parameters
#            has a size (2*b_radius+1)^2 
#
# path_fit: String with the adress where saving the txt file with the regression parameters.
#
# file_out1: String with the adress where saving the image file with the slope.
#
# file_out2: String with the adress where saving the image file with the slope.
#
# sigI : Error in the dimension x of the measures (Index)
#
# sigT : Error in the dimension y of the measures (Temperature)
#
# min_slope, max_slope, min_intercept, max_intercept: Min and max values allowed for slopes and intercepts.
#
# Exemple:
# path_index = '/tmp_user/ldota214h/cgranero/CATUT_unmixing/VNIR-SWIR/4_Index-Estimation/080628/Index_080628_20m.bsq'
# path_temperature = '/tmp_user/ldota214h/cgranero/CATUT_unmixing/TIR/4_TES/Outputs_080628_20m/LST_4_bandes_jour_satellite_20m_bruite.bsq'
# min_T = 285
# min_slope=-300
# max_slope=300
# min_intercept=270
# max_intercept=350
# sigI=0.01
# sigT=1
# path_fit = '/tmp_user/ldota214h/cgranero/CATUT_unmixing/TsHARP-DISTRAD/Scale_Inv/Fit2_NDVI_T_080628.txt'
# file_out1='/tmp_user/ldota214h/cgranero/CATUT_unmixing/AATPRK/Scale_Inv/Step1_Fits_080628/Slope_40m_NDVI.bsq'
# file_out2='/tmp_user/ldota214h/cgranero/CATUT_unmixing/AATPRK/Scale_Inv/Step1_Fits_080628/Intercept_40m_NDVI.bsq'
###
######
###########
############### C. Granero-Belinchon; ONERA, Toulouse; 01/2019

	import numpy as np
	from osgeo import gdal, gdalconst
	from scipy.stats import linregress
	import matplotlib.pyplot as plt
	from ltsfit.lts_linefit import lts_linefit
	
# "sys.stdout" lines prevent lts_linefit from printing in the console  
	import sys
	import os

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
	
# Two different linear regressions
	sigI1=np.full_like(I, sigI)
	sigT1=np.full_like(I, sigT)
	fit2 = lts_linefit(I, T, sigI1, sigT1, pivot=np.median(I))
	
	global_fit=np.zeros(2)
	global_fit[0]=fit2.ab[0]
	global_fit[1]=fit2.ab[1]
	
	local_fit=np.zeros(2)
#####################################################################
################### WINDOW DEFINITION , #############################
########### AND LINEAR REGRESSION FOR EACH WINDOW ###################

	a1 = np.zeros((rows, cols))
	a0 = np.zeros((rows, cols))

	for irow in range(b_radius,rows-b_radius):
		print("irow=",irow)
		print("icol=",end =" ")
		for icol in range(b_radius,cols-b_radius):
			print(icol,end =" ")
			Tw = Temp[irow-b_radius:irow+b_radius+1,icol-b_radius:icol+b_radius+1]
			Iw = index[irow-b_radius:irow+b_radius+1,icol-b_radius:icol+b_radius+1]
			Tw = Tw.flatten()
			Iw = Iw.flatten()
			mask=np.greater(Tw,min_T) # We eliminate the background pixels and those with a temperature under 285K
			Tw=Tw[mask]
			Iw=Iw[mask]
			NaN=np.isfinite(Iw) # We look for nans in the index and we eliminate those pixels from the index and the temperature
			Iw=Iw[NaN]
			Tw=Tw[NaN]
			sigI2=np.full_like(Iw, sigI)
			sigT2=np.full_like(Iw, sigT)
			try:
				if len(Tw) > 15:
					sys.stdout = open(os.devnull, "w")
					fit=lts_linefit(Iw,Tw,sigI2,sigT2,pivot=np.median(Iw))
					sys.stdout = sys.__stdout__
	
					local_fit[0]=fit.ab[0]
					local_fit[1]=fit.ab[1]
					a1[irow,icol]=local_fit[1]
					a0[irow,icol]=local_fit[0]
				if len(Tw) <= 15:
					a1[irow,icol]=global_fit[1]
					a0[irow,icol]=global_fit[0]
			except:
					print('#####',irow,icol,'#####')
					a1[irow,icol]=global_fit[1]
					a0[irow,icol]=global_fit[0]
		print(" ")
	
	for irow in range(0,b_radius):
		for icol in range(cols):			
			a1[irow,icol]=global_fit[1]
			a0[irow,icol]=global_fit[0]
			
	for irow in range(rows-b_radius,rows):
		for icol in range(cols):			
			a1[irow,icol]=global_fit[1]
			a0[irow,icol]=global_fit[0]
			
	for icol in range(0,b_radius):
		for irow in range(rows):			
			a1[irow,icol]=global_fit[1]
			a0[irow,icol]=global_fit[0]
			
	for icol in range(cols-b_radius,cols):
		for irow in range(rows):			
			a1[irow,icol]=global_fit[1]
			a0[irow,icol]=global_fit[0]
			
	
	mask=np.where(a1>max_slope)	
	mask_neg=np.where(a1<min_slope)
	a1[mask]=max_slope
	a1[mask_neg]=min_slope

	mask=np.where(a0>max_intercept)	
	mask_neg=np.where(a0<min_intercept)
	a0[mask]=max_intercept
	a0[mask_neg]=min_intercept
	
###################################################################
########### SAVING RESULTS ########################################
###################################################################			
				
#1
	driver = dataset_Temp.GetDriver()
	outDs = driver.Create(file_out1, int(cols), int(rows), 1, gdal.GDT_Float32)

	outBand = outDs.GetRasterBand(1)
	outBand.WriteArray(a1, 0, 0)
	outBand.FlushCache()
	
#2
	driver = dataset_Temp.GetDriver()
	outDs = driver.Create(file_out2, int(cols), int(rows), 1, gdal.GDT_Float32)

	outBand = outDs.GetRasterBand(1)
	outBand.WriteArray(a0, 0, 0)
	outBand.FlushCache()	
	
			
	np.savetxt(path_fit,(global_fit[0],global_fit[1]),fmt='%3.6f')	
		
	print('Fit Done')
	
	return	

def linear_fit_robust(path_index, path_temperature, min_T, por, path_fit, plot, path_plot):
	
############## It has any sense???
#
# This function takes two images (path_index and path_temperature) and 
# computes the linear regression parameters $T = a1*I + a0$. It also 
# applies two masks on the data. The first mask eliminates all the 
# pixels with temperatures below min_T, and the second masks eliminates 
# all the pixels with index == nan. In addition, it performs a second 
# regression without taking into account the outliers of the first one.
#  	
############## Inputs/Outputs:
#
# path_index: String leading to the index image file. (.bsq, .img, .raw)
#             This file must contain only one band.
#
# path_temperature: String leading to the index image file. (.bsq, .img, .raw)	
#                   This file must contain only one band.
#
# min_T: Minimum expected temperature in the image. All the temerpature 
#        below this one are considered spurious measures and not taken into account in the 
#        regression.	
#
# por : Pourcentage of suppressed outliers between the first and the second regressionspor.
#
# path_fit: String with the adress where saving the txt file with the regression parameters.
#
# plot: integer. = 0 if we don't want to obtain a plot, =1 if we want to obtain a plot. 
#
# path_plot: String with the adress where saving the plot of T vs I and the linear regression.
#
# Exemple:
# path_index = '/tmp_user/ldota214h/cgranero/CATUT_unmixing/TEST_Mod/Index/NDBI_080628.img'
# path_temperature = '/tmp_user/ldota214h/cgranero/CATUT_unmixing/TIR/4_TES/Outputs_080628_40m/LST_4_bandes_jour_satellite_40m_bruite.bsq'
# min_T = 285
# por =15 # 15%
# path_fit = '/tmp_user/ldota214h/cgranero/CATUT_unmixing/TsHARP-DISTRAD/Scale_Inv/Fit2_NDVI_T_080628.txt'
# plot =1
# path_plot = '/tmp_user/ldota214h/cgranero/CATUT_unmixing/TsHARP-DISTRAD/Scale_Inv/Figs_080628/NDVI_vs_T_080628.png'

################ Needed packages #### 
# import numpy as np
# import matplotlib.pyplot as plt
# from osgeo import gdal, gdalconst
# from scipy.stats import linregress
#
###
######
###########
############### C. Granero-Belinchon; ONERA, Toulouse; 06/2019

	from osgeo import gdal
	import numpy as np
	from scipy.stats import linregress
	import matplotlib.pyplot as plt
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
	
# Two different linear regressions

	fit = np.polyfit(I,T,1)
	fit_fn = np.poly1d(fit) 
	fit2=linregress(I,T)
	
########### RESIDUAL ESTIMATION AND OUTLIERS ELIMINATION #######

	T_2 = fit2[1]+fit2[0]*I
	
	# Residual estimation
	R = abs(T - T_2)  
	
	# We order the Residuals by increasing order
	Rx = np.argsort(R)
	
	# We eliminate the por % of pixels with higher Residuals
	por_2 = int(np.round(por*len(Rx)/100))
	Rx_2= Rx[0:len(Rx)-por_2]
	
	# We create the new T and I vectors without the outliers pixels
	
	TT = T[Rx_2]
	II = I[Rx_2]
	
	# Two different linear regressions

	fit = np.polyfit(II,TT,1)
	fit_fn_2 = np.poly1d(fit) 
	fit2=linregress(II,TT)
	
###################################################################
########### SAVING RESULTS ########################################
###################################################################
				
	np.savetxt(path_fit,(fit2[0],fit2[1],fit2[2],fit2[3],fit2[4]),fmt='%3.6f')	

###################################################################
########### PLOT ##################################################
###################################################################

	if plot == 1: 
		
		plt.figure(1)
		plt.scatter(I,T,s=1,c='blue')
		plt.scatter(II,TT,s=1,c='green')
		plt.plot(I,fit_fn(I),'--k')
		plt.plot(II,fit_fn_2(II),'--r')
		plt.xlabel('I')
		plt.ylabel('T')
		plt.grid()
		plt.savefig(path_plot)
		plt.close()
		
		print('Fit Done')
		
	else:
		
		print('Fit Done')
	
	return
