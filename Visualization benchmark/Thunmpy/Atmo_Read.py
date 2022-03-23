def Atmo_Read(Path,*args):

#ATMO_READ Python Code that read atmospheric terms from MODTRAN tape7 file.
#   Atmo_Read(Path) read from MODTRAN tape7 file the atmospheric
#   terms that is estimated by MODTRAN. There are 4 terms :
#       * Tup           : the upwelling atmospheric transmitance
#       * LatmUp        : the upwelling atmospheric radiance
#       * LatmDown      : the downwelling atmospheric radiance
#       * LatmDownTup   : the downwelling atmospheric radiance weighted by
#       the upwelling transmittance.
#   Input :
#       - Path      : Path pointing to the .txt of tape7 created by MODTRAN
#   Output :
#       - Atmo      : Structure containing the 4 atmospheric terms
#       with the MODTRAN wavelength (.Tup_Raw, .LatmUp_Raw, .LatmDown_Raw,
#       .LatmDownTup_Raw).
#   Optional Input :
#       - 'Calib'   : Structure containing the sensor calibration.
#       With this option, the atmospheric terms is interpolated using the
#       sensor calibration. Then, the "Atmo" structure contains the inter-
#       polated atmospheric terms (.Tup, .LatmUp, .LatmDown, .LatmDownTup).
#       If not inquire, there is no interpolation of the atmospheric terms.
# 
#   Manuel Cubero-Castan
#   $Revision: 1.0 $  $Date: 2014/07/09$
##
###
######
######### Python Version of Cubero-Castan Matlab code
###########
############### C. Granero-Belinchon; ONERA, Toulouse; 06/2019

	import numpy as np
	import sys
	sys.path.append("/tmp_user/ldota214h/cgranero/Codes_Manu/function")
	from Calib_Run import Calib_Run
	### Initialization
	
	nargin=len(args)+1
	
	    ### Error test
	if np.remainder(nargin,2) != 1 or nargin < 1:
	    print('Correct use of input : ( Image, ''Param'', Param)')
	    ### Parameters
	OptCalib = 0       # Calibration parameters (1 if interpolation, 
	                        #   0 otherwise).
	for ind in range(0,int((nargin-1)/2)):
		if args[ind]== 'Calib':
			Calib = args[ind+1]
			OptCalib = 1 
	### Read of tape7
	fid=open(Path,'r')
	for i in range(0,11): 
		fid.readline().rstrip()
	Param=np.loadtxt(fid)    
	#Param = Param[:,0:len(Param)-1]
	NbOnde = Param.shape[0]
	fid.close()
	
	Freq=np.zeros((NbOnde,1))
	Freq=10**4/Param[:,0]
	LatmUpRaw=np.zeros((NbOnde,1))              
	LatmDownRaw=np.zeros((NbOnde,1))
	LatmDownTupRaw=np.zeros((NbOnde,1))        
	TUpRaw=np.zeros((NbOnde,1))
	# http://modtran5.com/faqs/index.html : Indication Output tp7
	# http://twiki.cis.rit.edu/twiki/pub/Main/RadianceConversion/radconvert.pdf
	LatmUpRaw=Param[:,0]*Param[:,0]*(Param[:,2]+Param[:,5])
	TUpRaw=Param[:,1]
	LatmDownTupRaw = Param[:,0]*Param[:,0]*Param[:,7]
	LatmDownRaw=Param[:,0]*Param[:,0]*Param[:,7]/Param[:,1]
	
	### Remove all the terms Inf or Nan
	Ind_nan = np.isnan(LatmDownRaw)
	Freq = Freq[~Ind_nan]
	LatmUpRaw = LatmUpRaw[~Ind_nan]
	TUpRaw = TUpRaw[~Ind_nan]
	LatmDownTupRaw = LatmDownTupRaw[~Ind_nan]
	LatmDownRaw = LatmDownRaw[~Ind_nan]
	Ind_inf = np.isinf(LatmDownRaw)
	Freq = Freq[~Ind_inf]
	LatmUpRaw = LatmUpRaw[~Ind_inf]
	TUpRaw = TUpRaw[~Ind_inf]
	LatmDownTupRaw = LatmDownTupRaw[~Ind_inf]
	LatmDownRaw = LatmDownRaw[~Ind_inf]
	
	Atmo=np.zeros((4,2,len(Freq)))
	
	### Initialization of the "Atmo" structure
	Atmo[0] = [Freq, TUpRaw]         #Tup_Raw
	Atmo[1] = [Freq, LatmUpRaw]      #LatmUp_Raw
	Atmo[2] = [Freq, LatmDownRaw]    #LatmDown_Raw
	Atmo[3] = [Freq, LatmDownTupRaw] #LatmDownTup_Raw 
	
	Atmo_temporal = np.zeros((4,2,len(Freq)-1))
	
	Atmo_temporal[0]=Atmo[0][:,0:np.shape(Atmo[0])[1]-1]
	Atmo_temporal[1]=Atmo[1][:,0:np.shape(Atmo[1])[1]-1]
	Atmo_temporal[2]=Atmo[2][:,0:np.shape(Atmo[2])[1]-1]
	Atmo_temporal[3]=Atmo[3][:,0:np.shape(Atmo[3])[1]-1]
	
	Atmo=Atmo_temporal
	
	### Estimation of the interpolated atmospheric terms
	if OptCalib==1:
		
		Atmo2=np.zeros((4,2,len(Calib)))
		
		Atmo2[1] = Calib_Run(Calib,Atmo[1])
		Atmo2[0] = Calib_Run(Calib,Atmo[0])
		Atmo2[3] = Calib_Run(Calib,Atmo[3])
		Atmo2[2] = Calib_Run(Calib,Atmo[2])

		Atmo=Atmo2
		
	return Atmo
