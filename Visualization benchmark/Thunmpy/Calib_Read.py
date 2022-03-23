def Calib_Read(Path,*args):
#Calib_Read Python Code that read sensor calibration file.
#   Calib_Read(Path) read the calibration file pointed by the "Path" string.
#   Input :
#       - Path      : Path pointing to the .txt of the calibration file.
#       The .txt should be write as :
# --------------------
#   1- N, the number of bands (int)
#       2- Name of first band (string)
#       3- Number of values for the first band (int)
#       4- Table of the values of the sensor for each wavelength (µm). 
#       (float float)
# 
#   --> The item 2-, 3- and 4- is repeated N times.
# --------------------
#   Output :
#       - Calib     : Table containing in each raw a sructure
#       composed of 3 terms.
#           *   .Name   = Spectral band's name
#           *   .Value  = Matrix of the calibration values (lambda * %)
#           *   .Mean   = Mean of the spectral band (lambda)
#   Optional Input :
#       - 'Print'   : Option to print the results of the cailbration ( 1 ).
#       If not inquire, 'Print' is set to 0.
#       - 'Noise'   : Option to estimate the noise level for each band. If
#       inquire, the optional input is a path linking to a .txt file
#       containing the value of the band. If there is no value for each
#       bands, the Calib_Read function interpolated the value. The .txt
#       is structured as :
# --------------------
#   1- Type of noise (string) (SNR or NeDT)
#       2- Table of the values of the noise for each wavelength (µm). 
#       (float float)
# --------------------
#       A 4th item is added on the 'Calib' structure :
#           *   .SNR    = Signal-to-Noise Ratio (SNR) value.
#       If not inquire, no noise is set.
# 
#   Manuel Cubero-Castan
#   $Revision: 1.0 $  $Date: 2014/07/09$
##
###
######
######### Python Version of Cubero-Castan Matlab code
###########
############### C. Granero-Belinchon; ONERA, Toulouse; 06/2019

# Initialisation
	import numpy as np
	from scipy import interpolate
	import matplotlib.pyplot as plt
	import sys
	sys.path.append("/tmp_user/ldota214h/cgranero/Thunmpy_26-06-2019")
	from trust import Lcn

	nargin=len(args)+1
	
    ### Error Test
	if np.remainder(nargin,2) != 1 or nargin < 1:
		print('Correct use of input : ( Image, ''Param'', Param)');
	### Parameters
	OptPrint = 0           # Print of results
	Noise = 0              # SNR for each wavelength
	for ind in range(0,int((nargin-1)/2)):
		if args[2*ind]=='Print':
			OptPrint = args[2*ind+1]
			#print('OptPrint')
		elif args[2*ind]=='Noise':
			Noise = args[2*ind+1]
			#print(Noise)

# Reading of the file containing the filters
	File_Sens_Calib=open(Path,'r')

	tmp_nbr=np.loadtxt(File_Sens_Calib, max_rows=1)
	tmp_nbr=int(tmp_nbr)
	Calib=np.zeros((tmp_nbr,4), dtype=np.object)
	Calib[:]={} 
	Wave=np.zeros((tmp_nbr,1))
	for band in range(0,len(Calib)):
		Calib[band][0] = np.loadtxt(File_Sens_Calib,dtype='str',max_rows=1)    # Calib[band][0]== Name
		tmp_nbr=np.loadtxt(File_Sens_Calib,max_rows=1)
		Calib[band][1] = np.loadtxt(File_Sens_Calib,usecols=(0,1),max_rows=int(tmp_nbr))  # Calib[band][1]== Value
		Calib[band][2] = sum( Calib[band][1][:,0] * Calib[band][1][:,1])/sum(Calib[band][1][:,1]) # Calib[band][2]== Mean
		Wave[band] = Calib[band][2]

	File_Sens_Calib.close()
	
# Reading of the file containing the noise
	if Noise!=0:
		File_Noise  = open(Noise,'r')
		types       = np.loadtxt(File_Noise,dtype='str',max_rows=1)
		tmp         = np.loadtxt(File_Noise,usecols=(0,1))
		File_Noise.close()
		Noise       = np.zeros((tmp.shape[1],tmp.shape[0]+2))
		Noise[:,1:len(Noise[0])-1]    = tmp.T
		Noise[:,0]          = [1,tmp[1,1]]
		Noise[:,len(Noise[0])-1]        = [30, tmp[1,len(tmp[0])-1]]
		Noise = interpolate.interp1d(Noise[0,:],Noise[1,:])
		Noise = Noise(Wave)
		if types=='NeDT':
			for band in range(0, len(Wave)):
				Calib[band][3] = 20*np.log10(Lcn(Wave[band],300) /(Lcn(Wave[band],300+Noise[band]) - Lcn(Wave[band],300-Noise[band]))) # Calib[band][3]== SNR
		elif types=='SNR':
			for band in range(0, len(Wave)):
				Calib[band][3] = 20*np.log10(Noise[band])
					
# Plot of the sensor calibration
	if OptPrint == 1:
		cmap = color=plt.cm.hsv(np.arange(1,len(Calib)+1)/len(Calib))
		Leg1 = np.zeros((len(Calib),1))
		Leg2 = np.zeros((len(Calib),1),dtype=np.object)
		
		plt.figure()
		for band in range(1,len(Calib)):
			plt.plot(Calib[band][1][:,0],100*Calib[band][1][:,1],color=(tuple(cmap[band,:])))
			Leg2[band] = Calib[band][0]
		#plt.legend(Leg2[],'Location','EastOutside')
		plt.title('Sensor Calibrating')
		plt.xlabel('Wavelength (\mum)')    
		plt.ylabel('Proportion (%)')

	return Calib
