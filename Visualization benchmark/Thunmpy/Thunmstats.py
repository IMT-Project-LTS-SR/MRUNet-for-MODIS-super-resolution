def unm_stats(path_temperature, path_unm, min_T, path_file, path_file1, path_file2, plot, path_plot, path_plot1, path_plot2):
	
# Comments: This code can only be runned if we have a measured temperature to compare with the unmixed one.	
	
# Inputs/Outputs:
#
# path_unm: String leading to the unmixed temeprature image file. (.bsq, .img, .raw)
#             This file must contain only one band.
#
# path_temperature: String leading to the temperature image file. (.bsq, .img, .raw)	
#                   This file must contain only one band.
#
# min_T: Minimum expected temperature in the image. All the temerpature 
#        below this one are considered spurious measures and not taken into account in the 
#        regression.	
#
# path_file: String indicating the file (.txt) where saving rmse and mbe
#
# path_file1: String indicating the file (.txt) where saving ttest and p-value between unm temp and measured temp
#
# path_file2: String indicatinf the file (.txt )where saving cross corr between measure temp and unmixed temp.
#
# plot: integer. = 0 if we don't want to obtain a plot, =1 if we want to obtain a plot. 
#
# path_plot (1 and 2): String with the adress where saving the histograms of T and T unmixed.
#
# Exemple:
# path_unm = '/tmp_user/ldota214h/cgranero/CATUT_unmixing/TsHARP-DISTRAD/Scale_Inv/Step3_Images_080628/T_unmixed_final_20m_from40m_jourNDVI.bsq'
# path_temperature = '/tmp_user/ldota214h/cgranero/CATUT_unmixing/TIR/4_TES/Outputs_080628_20m/LST_4_bandes_jour_satellite_20m_bruite.bsq'
# min_T = 285
# path_file = "Output.txt"
# path_file1 = "Output_ttest_.txt"
# path_file2 = "Output_corr.txt"
# plot =1
# path_plot = '/tmp_user/ldota214h/cgranero/CATUT_unmixing/TsHARP-DISTRAD/Scale_Inv/Step4_Images_080628/Hist_T20m_080628_jour.png'
# path_plot1 = '/tmp_user/ldota214h/cgranero/CATUT_unmixing/TsHARP-DISTRAD/Scale_Inv/Step4_Images_080628/Hist_Tunm_40m_NDVI_080628_jour.png'
# path_plot2 = '/tmp_user/ldota214h/cgranero/CATUT_unmixing/TsHARP-DISTRAD/Scale_Inv/Step4_Images_080628/Hist_delta_40m_NDVI_080628_jour.png'


	import numpy as np
	import matplotlib.pyplot as plt
	from osgeo import gdal, gdalconst
	from scipy import stats

####################################################################
################ LOADING DATA ######################################
####################################################################

################### Temperature #####################################

# Reading temperature data
	TempFile = path_temperature
	dataset_Temp = gdal.Open(TempFile)

	cols_t = dataset_Temp.RasterXSize #Spatial dimension x
	rows_t = dataset_Temp.RasterYSize #Spatial dimension y

# Read as array
	Temp= dataset_Temp.ReadAsArray(0, 0, cols_t, rows_t).astype(np.float)
	
# Processing for plots	
	T=Temp.flatten()
	T_great_mask=np.greater(T,min_T) # We eliminate the background pixels and those with a temperature under 285K
	T=T[T_great_mask]

################### UNMIXED Temperature ###########################	

#datasets
	data_Temp_unm = gdal.Open(path_unm)

	cols = data_Temp_unm.RasterXSize #Spatial dimension x
	rows = data_Temp_unm.RasterYSize #Spatial dimension y

# Read as array
	Temp_unm = data_Temp_unm.ReadAsArray(0, 0, cols, rows).astype(np.float)

# Processing for plots	
	T_unm=Temp_unm.flatten()
	T_flat=T_unm[T_great_mask]
	T_great_mask_flat=np.greater(T_flat,min_T)
	T_flat=T_flat[T_great_mask_flat]
	T=T[T_great_mask_flat]
	delta = T - T_flat

#####################################################################
#################### RESULTS ########################################
#####################################################################

	text_file = open(path_file, "w")
	text_file_2 = open(path_file1, "w")
	text_file_3 = open(path_file2, "w")
	
# Basic Stats
	rmse=np.std(delta)
	mean_delta=np.mean(delta)
		
# Correlation
	xcorr_T = np.corrcoef(T_flat,T)	
		
# T-test and p value
	ttest,pvalue=stats.ttest_ind(T,T_flat)
		
	print(rmse,mean_delta, file=text_file, end='\n')
	print(ttest,pvalue, file=text_file_2, end='\n')
	print(xcorr_T[0,1],xcorr_T[1,0], file=text_file_3, end='\n')

	text_file.close()
	text_file_2.close()
	text_file_3.close()

#####################################################################
#################### PLOTS ##########################################
#####################################################################

	if plot == 1:
		plt.figure()
		plt.hist(T,bins='auto',color='black')
		plt.ylabel('T_20m')
		plt.grid()
		plt.savefig(path_plot)
		plt.close()

		plt.figure()
		plt.hist(T,bins='auto',color='black')
		plt.hist(T_flat,bins='auto',color='red')
		plt.ylabel('T')
		plt.grid()
		plt.savefig(path_plot1)
		plt.close()
		
		plt.figure()
		plt.hist(delta,bins='auto',color='red')
		plt.ylabel('delta_T')
		plt.grid()
		plt.savefig(path_plot2)
		plt.close()

		print('Stats Done')
	else:
		
		print('Stats Done')
	
	return
