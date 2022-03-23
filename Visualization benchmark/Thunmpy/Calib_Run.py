def  Calib_Run(Calib, Emiss_In):
	
#CALIB_RUN Run the interpolation using the caracteristic of the sensor.
#   Emiss_New = Calib_Run(Calib, Emiss_Input) interpolate the vector
#   contained in "Emiss_Input" with the sensor calibration "Calib".
#   Input :
#       - Calib     : Table containing in each raw a sructure
#       composed of 3 or 4 terms.
#           *   .Name   = Spectral band's name,
#           *   .Value  = Matrix of the calibration values (lambda * %),
#           *   .Mean   = Mean of the spectral band (lambda),
#           *   .SNR (Optional) = Signal-to-Noise Ratio (SNR) value.
#       - Emiss_In  : The matrix containing the wavelength and the vector
#       that need to be interpolated. The vector is stored as this :
#       Emiss_In = [ Wavelength;
#                    Old_Vector  ]
#       Its size is [2 x Nband_Old].
#   Output :
#       - Emiss_Out : The matrix containing the wavelength of each sensor
#       bands and the interpolated vector. Its size is [2 x Nband_Sensor].
# 
#   Manuel Cubero-Castan
#   $Revision: 1.0 $  $Date: 2014/07/09$
##
###
######
######### Python Version of Cubero-Castan Matlab code
###########
############### C. Granero-Belinchon; ONERA, Toulouse; 06/2019

### Increase of the emissivity
# Preprocess test on emissivity

	from scipy import interpolate
	import numpy as np

	if Emiss_In.shape[1]==2:  
		Emiss_In=Emiss_In.T
	elif Emiss_In.shape[0]!=2: 
		Emiss_Out=np.nan 
		return Emiss_Out
	if sum(sum(np.isnan(Emiss_In)))!=0: 
		print('NaN in Input Emiss') 
		Emiss_Out=np.nan 
		return Emiss_Out
	    
	# Re-order emissivity
	Emiss_In=np.sort(Emiss_In,axis=0)
	Emiss_In=np.flip(Emiss_In,1)
	# Size increase of the emissivity
	Emiss_Old = np.zeros((2,len(Emiss_In[0])+2))
	Emiss_Old[:,0] = [Emiss_In[0,0],1]
	Emiss_Old[:,1:len(Emiss_Old[0])-1] = Emiss_In
	Emiss_Old[:,len(Emiss_Old[0])-1] = [Emiss_In[0,len(Emiss_In[0])-1],30]
	Emiss_Out=np.zeros((2,len(Calib)))
	for band in range(0,len(Calib)):
		Emiss_Out[0,band]=Calib[band][2]
		tmp = interpolate.interp1d(Calib[band][1][:,0],Calib[band][1][:,1])
		tmp_em=Emiss_Old[1,:]
		tmp_em2=Emiss_Old[0,:]
		tmp_em2 = tmp_em2[np.where(tmp_em >= np.amin(Calib[band][1][:,0]))] 
		tmp_em = tmp_em[np.where(tmp_em >= np.amin(Calib[band][1][:,0]))]
		tmp_em2 = tmp_em2[np.where(tmp_em <= np.amax(Calib[band][1][:,0]))] 
		tmp_em = tmp_em[np.where(tmp_em <= np.amax(Calib[band][1][:,0]))] 
		tmp = tmp(tmp_em)
		tmp[np.isnan(tmp)]=0
		if sum(tmp)!=0:
			Emiss_Out[1,band] = sum(tmp_em2*tmp)/sum(tmp)
		else:
			tmp=interpolate.interp1d(tmp_em,tmp_em2)
			tmp = tmp(Calib[band][1][:,0])
			Emiss_Out[1,band] = sum(tmp*Calib[band][1][2,:])/sum(Calib[band][1][2,:])
				
		if sum(np.isnan(Emiss_Out[1,:]))!=0:
			print('Nan Values for interpolation !!!\n')
	
	return Emiss_Out
