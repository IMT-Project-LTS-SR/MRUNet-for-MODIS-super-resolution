def trust_DN(Image_Day, Image_Night, Emiss_Day, Emiss_Night, T_mean_Day, T_mean_Night, Atmo_Day, Atmo_Night, *args):
	
#   TRUST_DN is a new version of TRUST mixing DAY and Night information
#
#	### MAIN DIFFERENCE WITH trust in lin 209
#
#
#	trust_DN(Image_Day, Image_Night, Emiss_Day, Emiss_Night, T_mean_Day, T_mean_Night, Atmo_Day, Atmo_Night) 
#	is estimating the abundances S_mix and the subpixel temperatures T_mix using the images
#   Image_Day, Image_Night, the emissivities Emiss_Day, Emiss_Night and the mean temperatures T_mean_Day, T_mean_Night. 
#	It also needs the atmospherical terms Atmo_Day and Atmo_Night.
#   Input : (Both for Day and Night)
#       - Image     : It is the at-sensor radiance in W.sr-1.m-2.�m-1. A 
#       2-dim matrix (npix * nband) or a 3-dim matrix (nlength * nwidth *
#       nband) where nlength * nwidth = npix the number of pixels.
#       - Emiss     : Emissivity of each classes of material composing the
#       scene. The size is (nmat * nband) where nmat is the number of
#       classes of material.
#       - T_mean    : The mean temperature of classes of material in K. It
#       is a vector of nmat elements.
#       - Atmo      : Structure composed by the atmospheric terms
#       (Tup, LatmDown, LatmUp).
#   Output :
#       - S_mix     : The abundances estimated by TRUST. Depending on the
#       size of the input Image, S_mix is a 2-dim matrix (npix * nmat) or
#       a 3-dim matrix (nlength * nwidth * nmat).
#       - T_mix_Day     : The subpixel temperature estimated by TRUST. There is
#       the same size than S_mix.
#       - T_mix_Night     : The subpixel temperature estimated by TRUST. There is
#       the same size than S_mix.
#   Optional Input :
#       - 'nmat_stop'   : Maximum number of material composing the mixed
#       pixels. If not inquire, nmat_stop = 3.
#       - 'gamma'       : Hyperparameter gamma for the minimization. If not
#       inquire, gamma = 0.
#   Optional Output :
#       "Error" is the RMSE of the reconstruction for all pixels. Depening on the size
#       of Image. "Error" is a 2-dim matrix (npix * nperms) or a 3-dim matrix (nlength *
#       nwidth * nperms). nperms is the number of possible permutations.
# 
#   Reference : M. Cubero-Castan et al., 'A physic-based unmixing method to
#   estimate subpixel temperatures on mixed pixels', TGRS 2015.
#   Manuel Cubero-Castan
#   Revision: 1.0 $  $Date: 2014/07/09$
##
###
######
######### Python Version of Cubero-Castan Matlab code
###########
############### C. Granero-Belinchon; ONERA, Toulouse; 06/2019

# We import the needed python libraries
	import numpy as np
	import scipy.optimize as opt
	from scipy.optimize import LinearConstraint
	import sys
	sys.path.append("/tmp_user/ldota214h/cgranero/Thunmpy/")
	from trust_DN import trust_cest_DN
	from trust import perms_Manu
	
	import warnings
	warnings.filterwarnings("ignore") ##ok<WNOFF>
	### Initialisation
	
	## We read the size of the image
	## We define the 2D size of the image for any 2D or 3D image. We also define the paramer Opt3D indicating 3D image if it's equal to 1.
	## If SizeIm.shape is not 2 or 3 dimensional then there is a problem.
	SizeIm_Day = Image_Day.shape
	SizeIm_Night = Image_Night.shape
	if SizeIm_Day!=SizeIm_Night:
		print('Error --- Day and Night images dont have the same shape')
		sys.exit()
	SizeIm=SizeIm_Day	
	if len(SizeIm) == 2:
		Size2D = SizeIm 
		Im2D_Day = Image.Day
		Im2D_Night = Image.Night 
		Opt3D = 0
	elif len(SizeIm) == 3:	             
		Size2D = [SizeIm[0]*SizeIm[1], SizeIm[2]]
		Im2D_Day = np.reshape(Image_Day,Size2D,order='F')
		Im2D_Night = np.reshape(Image_Night,Size2D,order='F')
		Opt3D = 1
	else:
		print('Size Image Problem !!!\n') 
		sys.exit()

	### Emiss should be a matrix M x Nmat (Bands times materials)
	if Emiss_Day.shape!=Emiss_Night.shape:
		print('Error --- Day and Night Emissivities dont have the same shape')
		sys.exit()
	if len(Emiss_Day.shape)==1:          # This is a test to avoid python dimensionality problems
		Emiss_Day=Emiss_Day[np.newaxis,:]
		Emiss_Night=Emiss_Night[np.newaxis,:]
	if len(Emiss_Day[0,:]) == Size2D[1]: 
		Emiss_Day = Emiss_Day.T # Transpose of a 2D vector
		Emiss_Night = Emiss_Night.T
	
    ### We define the number of materials Nmat as the dimension of Emiss wich is different to M (Size2D(2)=M=number of bands).
	NMat = len(Emiss_Day[0,:])

	### Finally, we arrange the Emiss and T_mean arrays to have Nmat as the second dimension.
	if T_mean_Day.shape!=T_mean_Night.shape:
		print('Error --- Day and Night Temperatures dont have the same shape')
		sys.exit()
	if len(T_mean_Day.shape)==1:         # This is a test to avoid python dimensionality problems
		T_mean_Day=T_mean_Day[np.newaxis,:]
		T_mean_Night=T_mean_Night[np.newaxis,:]
	if len(T_mean_Day[:,0]) == NMat: 
		T_mean_Day = T_mean_Day.T 
		T_mean_Night = T_mean_Night.T
	
	### Options
	### We define the different possibilities (varargin) arguments of TRUST function.
	### Major Parameters
	### NMat_Stop == Maximum number of materials per pixel ??
	### gamma == hyperparameter for second minimisation taking into account temperature differences
	### Minor parameters
    ### nargout == number of additional outputs
	gamma = 0      # It can take any value
	NMat_Stop = 3  # It should be limited to 3. Cubero-Castan suggestion
	nargout=0
	nargin=len(args)+4
	
	for ind in range(0,nargin-5,2):
		### Major parameters
		if args[ind] == 'nmat_stop': 
			NMat_Stop = args[ind+1]
		elif args[ind] ==	'gamma': 
			gamma = args[ind+1]	
	    ### Minor parameters
		elif args[ind] == 'nargout': 
			nargout = args[ind+1]
	
	### Initialisation

    ### Permutation
    ### P is a empty matrix of dimension (Nmat,1)
    ### Each element of the matrix P is defined as perms_Manu(i,3), which defines the possible combinations of materials with a maximum of three different materials per pixel.
	P= perms_Manu(NMat,NMat_Stop) 
	nperms=len(P)
	for l in range(0,nperms):
		P_tmp=P[l]
		P_tmp[:]=[x-1 for x in P_tmp]	
	Kl = np.arange(0,NMat)   # As it will be used as index we start in zero.
	
	### Outputs initialization
	### We initialize the result arrays S_mix (abondances =  pixels times materials matrix) and T_mix (temperatures = pixels times materials matrix)
	S_mix_Day = np.zeros((Size2D[0],NMat))
	S_mix_Night = np.zeros((Size2D[0],NMat))     
	T_mix_Day = np.zeros((Size2D[0],NMat))
	T_mix_Day[:]=np.nan
	T_mix_Night = np.zeros((Size2D[0],NMat))
	T_mix_Night[:]=np.nan
	
	### Else Err == a nan matrix of dimension (npixels x  possible combinations of three NMat materials)    
	Err = np.zeros((Size2D[0],nperms))
	Err[:]=np.nan
		
	### Non nan pixels search
	### We look for the pixels where at least in one band have nan values. Ind == all the other pixels
	### DAY-NIGHT
	tmp_d = np.sum(Im2D_Day,axis=1)  
	tmp_n = np.sum(Im2D_Night,axis=1) 
	tmp = tmp_d+tmp_n
	tmp=tmp[np.newaxis,:]
	tmp=tmp.T
	Ind = np.where(~np.isnan(tmp))
	Ind = Ind[0]
	### Loop over the non nan pixels
	### i goes from 1 to the number of non nan pixels in the image. x is the coordinates of the non nan pixels
	x=np.zeros((2,1))
	for i in range(0,len(Ind)):    
		x = Ind[i]
		x=x.astype(int)	
		print('x=',x)
		try:
			### Vectors' Initialisation
			D_Vect_Day = np.zeros((nperms,1))         # Reconstruction error
			D_Vect_Day[:] = np.nan
			D_Vect_Night = np.zeros((nperms,1))         # Reconstruction error
			D_Vect_Night[:] = np.nan
			T_Vect_Day = np.zeros((nperms,NMat))      # Temperature vector
			T_Vect_Day[:] = np.nan
			T_Vect_Night = np.zeros((nperms,NMat))      # Temperature vector
			T_Vect_Night[:] = np.nan
			S_Vect_Day = np.zeros((nperms,NMat))    # Abondance vector
			S_Vect_Night = np.zeros((nperms,NMat))    # Abondance vector
			
			for l in range(0,nperms):
				#print('l=',l)
				Cc = P[l]
				#print('Cc=',Cc)  
				N_cc = len(Cc)
				## Minimization S with CEST
				S_2 = opt.minimize(lambda S: trust_cest_DN(S, Im2D_Day[x,:], Emiss_Day[:,Cc], T_mean_Day[:,Cc], Atmo_Day, 'gamma', 0,'nargout',0), (1/N_cc)*np.ones((N_cc)), constraints = LinearConstraint(np.ones((N_cc)), 1, 1), bounds=opt.Bounds(0.01, 0.99), method='trust-constr')
				[DD,TT] = trust_cest_DN(S_2.x, Im2D_Day[x,:], Emiss_Day[:,Cc], T_mean_Day[:,Cc], Atmo_Day, 'gamma', gamma,'nargout',1)
				S_Vect_Day[l,Cc] = S_2.x
				D_Vect_Day[l]=DD
				T_Vect_Day[l,Cc]=TT
				if sum(S_Vect_Day[l,Cc]<0)>0: 
					D_Vect_Day[l]=np.nan
				S_2 = opt.minimize(lambda S: trust_cest_DN(S, Im2D_Night[x,:], Emiss_Night[:,Cc], T_mean_Night[:,Cc], Atmo_Night, 'gamma', 0,'nargout',0), (1/N_cc)*np.ones((N_cc)), constraints = LinearConstraint(np.ones((N_cc)), 1, 1), bounds=opt.Bounds(0.01, 0.99), method='trust-constr')
				[DD,TT] = trust_cest_DN(S_2.x, Im2D_Night[x,:], Emiss_Night[:,Cc], T_mean_Night[:,Cc], Atmo_Night, 'gamma', gamma,'nargout',1)
				S_Vect_Night[l,Cc] = S_2.x
				D_Vect_Night[l]=DD
				T_Vect_Night[l,Cc]=TT
				if sum(S_Vect_Night[l,Cc]<0)>0: 
					D_Vect_Night[l]=np.nan
	        ### Search of the best reconstruction
	        ### MAIN DIFFERENCE WITH trust (NO DAY_NIGHT)
			D_Vect=D_Vect_Day+D_Vect_Night
			Num=np.argmin(D_Vect,0) # Array of indices with minimum values along axis 0
			T_mix_Day[x,:]  = T_Vect_Day[Num,:]
			T_mix_Night[x,:]  = T_Vect_Night[Num,:]
			S_mix_Day[x,:]  = S_Vect_Day[Num,:]
			S_mix_Night[x,:]  = S_Vect_Night[Num,:]
			Err[x,0:nperms]  = D_Vect[:].T
		except:
			print('# = %d\n',i) 
			T_mix_Day[x,:]=np.nan 
			T_mix_Night[x,:]=np.nan
			S_mix_Day[x,:]=np.nan  ##ok<CTCH>
			S_mix_Night[x,:]=np.nan
	### Reshape of the image to the original shape
	### The outputs of the function are S_mix and T_mix with the shapes of the input image times NMat (number of materials).
	if Opt3D==1: 
		S_mix_Day = np.reshape(S_mix_Day,(int(SizeIm[0]),int(SizeIm[1]),int(NMat)),order='F')   
		S_mix_Night = np.reshape(S_mix_Night,(int(SizeIm[0]),int(SizeIm[1]),int(NMat)),order='F') 
		T_mix_Day = np.reshape(T_mix_Day,(SizeIm[0],SizeIm[1],NMat),order='F') 
		T_mix_Night = np.reshape(T_mix_Night,(SizeIm[0],SizeIm[1],NMat),order='F')
		Err = np.reshape(Err,(SizeIm[0],SizeIm[1],nperms),order='F')
	if NMat_Stop == 1: 
		if Opt3D==1: 
			S_mix = np.reshape(S_mix,(Size2D[0],NMat),order='F')
		C = np.zeros((1,Size2D[1]))
		for b in range(0,NMat): 
			C[:] = C[:] + b * (S_mix[:,b]==1)
		C[C==0]=np.nan
		if Opt3D==1: 
			S_mix = np.reshape(C,(SizeIm[0],SizeIm[1]),order='F') 
		else: 
			S_mix=C
    ### If we ask for more than two outputs, we can obtain the error per pixel.
	if nargout>4:                                
		varargout = Err
		return S_mix_Day, S_mix_Night, T_mix_Day,T_mix_Night,varargout
	return S_mix_Day, S_mix_Night, T_mix_Day,T_mix_Night
	
def trust_DN2(Image_Day, Image_Night, Emiss_Day, Emiss_Night, T_mean_Day, T_mean_Night, Atmo_Day, Atmo_Night, *args):
	
#   TRUST_DN is a new version of TRUST mixing DAY and Night information
#
#	### MAIN DIFFERENCE WITH trust in lin 209
#
#
#	trust_DN(Image_Day, Image_Night, Emiss_Day, Emiss_Night, T_mean_Day, T_mean_Night, Atmo_Day, Atmo_Night) 
#	is estimating the abundances S_mix and the subpixel temperatures T_mix using the images
#   Image_Day, Image_Night, the emissivities Emiss_Day, Emiss_Night and the mean temperatures T_mean_Day, T_mean_Night. 
#	It also needs the atmospherical terms Atmo_Day and Atmo_Night.
#   Input : (Both for Day and Night)
#       - Image     : It is the at-sensor radiance in W.sr-1.m-2.�m-1. A 
#       2-dim matrix (npix * nband) or a 3-dim matrix (nlength * nwidth *
#       nband) where nlength * nwidth = npix the number of pixels.
#       - Emiss     : Emissivity of each classes of material composing the
#       scene. The size is (nmat * nband) where nmat is the number of
#       classes of material.
#       - T_mean    : The mean temperature of classes of material in K. It
#       is a vector of nmat elements.
#       - Atmo      : Structure composed by the atmospheric terms
#       (Tup, LatmDown, LatmUp).
#   Output :
#       - S_mix     : The abundances estimated by TRUST. Depending on the
#       size of the input Image, S_mix is a 2-dim matrix (npix * nmat) or
#       a 3-dim matrix (nlength * nwidth * nmat).
#       - T_mix_Day     : The subpixel temperature estimated by TRUST. There is
#       the same size than S_mix.
#       - T_mix_Night     : The subpixel temperature estimated by TRUST. There is
#       the same size than S_mix.
#   Optional Input :
#       - 'nmat_stop'   : Maximum number of material composing the mixed
#       pixels. If not inquire, nmat_stop = 3.
#       - 'gamma'       : Hyperparameter gamma for the minimization. If not
#       inquire, gamma = 0.
#   Optional Output :
#       "Error" is the RMSE of the reconstruction for all pixels. Depening on the size
#       of Image. "Error" is a 2-dim matrix (npix * nperms) or a 3-dim matrix (nlength *
#       nwidth * nperms). nperms is the number of possible permutations.
# 
#   Reference : M. Cubero-Castan et al., 'A physic-based unmixing method to
#   estimate subpixel temperatures on mixed pixels', TGRS 2015.
#   Manuel Cubero-Castan
#   Revision: 1.0 $  $Date: 2014/07/09$
##
###
######
######### Python Version of Cubero-Castan Matlab code
###########
############### C. Granero-Belinchon; ONERA, Toulouse; 06/2019

# We import the needed python libraries
	import numpy as np
	import scipy.optimize as opt
	from scipy.optimize import LinearConstraint
	import sys
	sys.path.append("/tmp_user/ldota214h/cgranero/Thunmpy/")
	from trust_DN import trust_cest_DN
	from trust import perms_Manu
	
	import warnings
	warnings.filterwarnings("ignore") ##ok<WNOFF>
	### Initialisation
	
	## We read the size of the image
	## We define the 2D size of the image for any 2D or 3D image. We also define the paramer Opt3D indicating 3D image if it's equal to 1.
	## If SizeIm.shape is not 2 or 3 dimensional then there is a problem.
	SizeIm_Day = Image_Day.shape
	SizeIm_Night = Image_Night.shape
	if SizeIm_Day!=SizeIm_Night:
		print('Error --- Day and Night images dont have the same shape')
		sys.exit()
	SizeIm=SizeIm_Day	
	if len(SizeIm) == 2:
		Size2D = SizeIm 
		Im2D_Day = Image.Day
		Im2D_Night = Image.Night 
		Opt3D = 0
	elif len(SizeIm) == 3:	             
		Size2D = [SizeIm[0]*SizeIm[1], SizeIm[2]]
		Im2D_Day = np.reshape(Image_Day,Size2D,order='F')
		Im2D_Night = np.reshape(Image_Night,Size2D,order='F')
		Opt3D = 1
	else:
		print('Size Image Problem !!!\n') 
		sys.exit()

	### Emiss should be a matrix M x Nmat (Bands times materials)
	if Emiss_Day.shape!=Emiss_Night.shape:
		print('Error --- Day and Night Emissivities dont have the same shape')
		sys.exit()
	if len(Emiss_Day.shape)==1:          # This is a test to avoid python dimensionality problems
		Emiss_Day=Emiss_Day[np.newaxis,:]
		Emiss_Night=Emiss_Night[np.newaxis,:]
	if len(Emiss_Day[0,:]) == Size2D[1]: 
		Emiss_Day = Emiss_Day.T # Transpose of a 2D vector
		Emiss_Night = Emiss_Night.T
	
    ### We define the number of materials Nmat as the dimension of Emiss wich is different to M (Size2D(2)=M=number of bands).
	NMat = len(Emiss_Day[0,:])

	### Finally, we arrange the Emiss and T_mean arrays to have Nmat as the second dimension.
	if T_mean_Day.shape!=T_mean_Night.shape:
		print('Error --- Day and Night Temperatures dont have the same shape')
		sys.exit()
	if len(T_mean_Day.shape)==1:         # This is a test to avoid python dimensionality problems
		T_mean_Day=T_mean_Day[np.newaxis,:]
		T_mean_Night=T_mean_Night[np.newaxis,:]
	if len(T_mean_Day[:,0]) == NMat: 
		T_mean_Day = T_mean_Day.T 
		T_mean_Night = T_mean_Night.T
	
	### Options
	### We define the different possibilities (varargin) arguments of TRUST function.
	### Major Parameters
	### NMat_Stop == Maximum number of materials per pixel ??
	### gamma == hyperparameter for second minimisation taking into account temperature differences
	### Minor parameters
    ### nargout == number of additional outputs
	gamma = 0      # It can take any value
	NMat_Stop = 3  # It should be limited to 3. Cubero-Castan suggestion
	nargout=0
	nargin=len(args)+4
	
	for ind in range(0,nargin-5,2):
		### Major parameters
		if args[ind] == 'nmat_stop': 
			NMat_Stop = args[ind+1]
		elif args[ind] ==	'gamma': 
			gamma = args[ind+1]	
	    ### Minor parameters
		elif args[ind] == 'nargout': 
			nargout = args[ind+1]
	
	### Initialisation

    ### Permutation
    ### P is a empty matrix of dimension (Nmat,1)
    ### Each element of the matrix P is defined as perms_Manu(i,3), which defines the possible combinations of materials with a maximum of three different materials per pixel.
	P= perms_Manu(NMat,NMat_Stop) 
	nperms=len(P)
	for l in range(0,nperms):
		P_tmp=P[l]
		P_tmp[:]=[x-1 for x in P_tmp]	
	Kl = np.arange(0,NMat)   # As it will be used as index we start in zero.
	
	### Outputs initialization
	### We initialize the result arrays S_mix (abondances =  pixels times materials matrix) and T_mix (temperatures = pixels times materials matrix)
	S_mix = np.zeros((Size2D[0],NMat))    
	T_mix_Day = np.zeros((Size2D[0],NMat))
	T_mix_Day[:]=np.nan
	T_mix_Night = np.zeros((Size2D[0],NMat))
	T_mix_Night[:]=np.nan
	
	### Else Err == a nan matrix of dimension (npixels x  possible combinations of three NMat materials)    
	Err = np.zeros((Size2D[0],2*nperms))
	Err[:]=np.nan
		
	### Non nan pixels search
	### We look for the pixels where at least in one band have nan values. Ind == all the other pixels
	### DAY-NIGHT
	tmp_d = np.sum(Im2D_Day,axis=1)  
	tmp_n = np.sum(Im2D_Night,axis=1) 
	tmp = tmp_d+tmp_n
	tmp=tmp[np.newaxis,:]
	tmp=tmp.T
	Ind = np.where(~np.isnan(tmp))
	Ind = Ind[0]
	### Loop over the non nan pixels
	### i goes from 1 to the number of non nan pixels in the image. x is the coordinates of the non nan pixels
	x=np.zeros((2,1))
	for i in range(0,len(Ind)):    
		x = Ind[i]
		x=x.astype(int)	
		print('x=',x)
		#try:
		### Vectors' Initialisation
		D_Vect_Day = np.zeros((nperms,1))         # Reconstruction error
		D_Vect_Day[:] = np.nan
		D_Vect_Night = np.zeros((nperms,1))         # Reconstruction error
		D_Vect_Night[:] = np.nan
		T_Vect_Day = np.zeros((nperms,NMat))      # Temperature vector
		T_Vect_Day[:] = np.nan
		T_Vect_Night = np.zeros((nperms,NMat))      # Temperature vector
		T_Vect_Night[:] = np.nan
		S_Vect_Day = np.zeros((nperms,NMat))    # Abondance vector
		S_Vect_Night = np.zeros((nperms,NMat))    # Abondance vector
		
		for l in range(0,nperms):
			#print('l=',l)
			Cc = P[l]
			#print('Cc=',Cc)  
			N_cc = len(Cc)
			## Minimization S with CEST
			S_2 = opt.minimize(lambda S: trust_cest_DN(S, Im2D_Day[x,:], Emiss_Day[:,Cc], T_mean_Day[:,Cc], Atmo_Day, 'gamma', 0,'nargout',0), (1/N_cc)*np.ones((N_cc)), constraints = LinearConstraint(np.ones((N_cc)), 1, 1), bounds=opt.Bounds(0.01, 0.99), method='trust-constr')
			[DD,TT] = trust_cest_DN(S_2.x, Im2D_Day[x,:], Emiss_Day[:,Cc], T_mean_Day[:,Cc], Atmo_Day, 'gamma', gamma,'nargout',1)
			S_Vect_Day[l,Cc] = S_2.x
			D_Vect_Day[l]=DD
			T_Vect_Day[l,Cc]=TT
			if sum(S_Vect_Day[l,Cc]<0)>0: 
				D_Vect_Day[l]=np.nan
			S_2 = opt.minimize(lambda S: trust_cest_DN(S, Im2D_Night[x,:], Emiss_Night[:,Cc], T_mean_Night[:,Cc], Atmo_Night, 'gamma', 0,'nargout',0), (1/N_cc)*np.ones((N_cc)), constraints = LinearConstraint(np.ones((N_cc)), 1, 1), bounds=opt.Bounds(0.01, 0.99), method='trust-constr')
			[DD,TT] = trust_cest_DN(S_2.x, Im2D_Night[x,:], Emiss_Night[:,Cc], T_mean_Night[:,Cc], Atmo_Night, 'gamma', gamma,'nargout',1)
			S_Vect_Night[l,Cc] = S_2.x
			D_Vect_Night[l]=DD
			T_Vect_Night[l,Cc]=TT
			if sum(S_Vect_Night[l,Cc]<0)>0: 
				D_Vect_Night[l]=np.nan
        ### Search of the best reconstruction
        ### MAIN DIFFERENCE WITH trust (NO DAY_NIGHT)
		D_Vect=np.concatenate((D_Vect_Day,D_Vect_Night),axis=0)
		Num=np.argmin(D_Vect,0) # Array of indices with minimum values along axis 0
		S_Vect=np.concatenate((S_Vect_Day,S_Vect_Night),axis=0)
		S_mix[x,:]=S_Vect[Num,:]
		Cc=np.where(S_mix[x,:]!=0)
		Cc=list(Cc[0])
		[DD,TT] = trust_cest_DN(S_mix[x,Cc], Im2D_Day[x,:], Emiss_Day[:,Cc], T_mean_Day[:,Cc], Atmo_Day, 'gamma', gamma,'nargout',1)
		T_mix_Day[x,Cc]  = TT
		[DD,TT] = trust_cest_DN(S_mix[x,Cc], Im2D_Night[x,:], Emiss_Night[:,Cc], T_mean_Night[:,Cc], Atmo_Night, 'gamma', gamma,'nargout',1)
		T_mix_Night[x,Cc]  = TT
		Err[x,0:2*nperms]  = D_Vect[:].T
		#except:
		#	print('# = %d\n',i) 
		#	T_mix_Day[x,:]=np.nan 
		#	T_mix_Night[x,:]=np.nan
		#	S_mix[x,:]=np.nan  ##ok<CTCH>
	### Reshape of the image to the original shape
	### The outputs of the function are S_mix and T_mix with the shapes of the input image times NMat (number of materials).
	if Opt3D==1: 
		S_mix = np.reshape(S_mix,(int(SizeIm[0]),int(SizeIm[1]),int(NMat)),order='F')   
		T_mix_Day = np.reshape(T_mix_Day,(SizeIm[0],SizeIm[1],NMat),order='F') 
		T_mix_Night = np.reshape(T_mix_Night,(SizeIm[0],SizeIm[1],NMat),order='F')
		Err = np.reshape(Err,(SizeIm[0],SizeIm[1],2*nperms),order='F')
	if NMat_Stop == 1: 
		if Opt3D==1: 
			S_mix = np.reshape(S_mix,(Size2D[0],NMat),order='F')
		C = np.zeros((1,Size2D[1]))
		for b in range(0,NMat): 
			C[:] = C[:] + b * (S_mix[:,b]==1)
		C[C==0]=np.nan
		if Opt3D==1: 
			S_mix = np.reshape(C,(SizeIm[0],SizeIm[1]),order='F') 
		else: 
			S_mix=C
    ### If we ask for more than two outputs, we can obtain the error per pixel.
	if nargout>4:                                
		varargout = Err
		return S_mix, T_mix_Day,T_mix_Night,varargout
	return S_mix, T_mix_Day,T_mix_Night

def trust_cest_DN(S_tmp,Lsens,Emiss2,Temp,Atmo,*args):
	
#   TRUST_CEST_DN Python Code of the Constrain Estimation of Subpixel Temperature with Normalization
#
#	### MAIN DIFFERENCE WITH trust_cest in line 352
#
#	trust_cest_DN(S_tmp, Lsens, Emiss, Temp, Atmo) is estimating the
#	error Diff when the subpixel temperature is retrieved. S_tmp is the
#	input abundance, Lsens the at-sensor radiance, Emiss the emissivity of
#	the classes of materials composing the mixed pixels, Temp the mean
#	temperature of the classes of materials and Atmo the atmospheric terms
#	structure.
# 
#   Input :
#       - S_tmp     : The input abundance. A vector of size (nmat * 1) 
#       where nmat is the number of classes of material.
#       - Lsens     : The input at-sensor radiance in W.sr-1.m-2.�m-1. A
#       vector of size (1 * nband) where nband is the number of spectral
#       bands.
#       - Emiss     : Emissivity of each classes of material composing the
#       pixel. The size is (nband * nmat).
#       - Temp      : The mean temperature of each classes of material. The
#       size is (nmat * 1).
#       - Atmo      : Structure composed by the atmospheric terms
#       (Tup, LatmDown, LatmUp).
#   Optional Input :
#       - 'gamma'   : Hyperparameter gamma for the minimization. If not
#       inquire, gamma = 0.
#       - 'noise_inv' : If the Noise Covariance matrix 'C' is known. If
#       it is not inquire, C = eye(nband).
#       - 'nargout' : number of additional outputs
#   Output :
#       - Diff      : The cost function linked the RMSE of the reconstruc-
#       tion and the physical constrain on temperature with the hyper-
#       parameter gamma.
#   Optional Output :
#       T_mix is the subpixel temperature in K.
#       CN is the Condition Number of the estimation, without dimension.
#       CRLB is the Cramer-Rao lower bound, in K�.
#
#   Reference : M. Cubero-Castan et al., 'An unmixing-based method for the
#   analysis of thermal hyperspectral images', ICASSP 2014.
#   Copyright 2014 ONERA - MCC PhD
#   Manuel Cubero-Castan
#   $Revision: 1.0 $  $Date: 2014/07/09$	
###
#####
####### 
######### Python Version of Cubero-Castan Matlab code
###########
############# Granero-Belinchon, Carlos;  ONERA, Toulouse; 06/2019

# We import the python libraries
	import numpy as np
	import scipy.linalg as la
	import sys
	sys.path.append("/tmp_user/ldota214h/cgranero/Thunmpy/")
	from trust import Lcn
	from trust import DLcn
	
	S_tmp=np.array(S_tmp)
	if len(S_tmp.shape)==1:          # This is a test to avoid python dimensionality problems
		S_tmp=S_tmp[:,np.newaxis]
	
	### Atmosphere terms
	Ld = Atmo[3][1,:] 
	Lu = Atmo[1][1,:] 
	Tu = Atmo[0][1,:]
	Wave = Atmo[0][0,:]  
	NWave = len(Wave) 
	if len(Emiss2.shape)==1:          # This is a test to avoid python dimensionality problems
		Emiss2=Emiss2[:,np.newaxis]
	NMat = len(Emiss2[0,:])
	
	### Study of varargin
	gamma = 0
	C = np.identity(NWave);
	
	nargin=len(args)+5
	
	for ind in range(0,nargin-5,2):
		if args[ind]=='gamma':
			gamma = args[ind+1]
		elif args[ind]=='noisecorr_inv':
			C = args[ind+1]
		elif args[ind]=='nargout':
			nargout = args[ind+1]
				
	### Build LcnMat & DLcnMat
	LcnMat = Lcn(Wave,Temp) 
	DLcnMat = DLcn(Wave,Temp)
	### Inversion Temperature (section 4.3.1 Cubero-Castan Thesis)
	Lsens_tmp = np.sum( Emiss2.T * LcnMat * (S_tmp*Tu) + (1-Emiss2.T) * (np.ones((NMat,1))*Ld) * S_tmp + (S_tmp*Lu),axis=0)
	B = (Lsens.T - Lsens_tmp.T)
	A = (np.transpose(Tu)*np.ones((NMat,1))) * np.transpose(Emiss2) * (np.ones((1,NWave))*S_tmp) * DLcnMat
	Mfish = np.matmul(np.matmul(A,C),A.T)
	temporal=la.solve(Mfish,np.matmul(np.matmul(A,C),B))
	T = Temp + temporal.T  
	
	### LsensNew estimation (with the temperature obtained in the above lines)
	LcnMat = Lcn(Wave,T)
	Lsens_tmp = np.sum( Emiss2.T * LcnMat * (S_tmp*Tu) + (1-Emiss2.T) * (np.ones((NMat,1))*Ld) * S_tmp + (S_tmp*Lu),axis=0)
	
	### Difference with the inputs (section 4.4.2 Cubero-Castan Thesis)
	#Diff = np.sqrt(np.mean((Lsens_tmp - Lsens)**2)) + gamma * np.sqrt(np.mean((T-Temp)**2))
	### MAIN DIFFERENCE WITH trust_cest (NO DAY_NIGHT)
	Diff = np.sqrt(np.mean(((Lsens_tmp - Lsens)/Lsens)**2)) + gamma * np.sqrt(np.mean(((T-Temp)/Temp)**2))
	### Study of varargout : 1 = Temperature, 2 = Condition Number
	if nargout != 0:
		varargout=np.zeros((nargout,len(T[:,0]),len(T[0,:])))
		if nargout >= 1:
			varargout[0] = T
		### Construction of the CN
		if nargout >= 2: 
			tmp2,tmp = la.eig(Mfish) 
			varargout[1] = max(tmp)/min(tmp)
		if nargout == 3: 
			varargout[2] = np.diag(inv(Mfish))
	
		return Diff,varargout 
		
	return Diff
