### Main TRUST
# Script pour réaliser le démixage via TRUST de manière supervisée.
import sys
sys.path.append("./Thunmpy")

from Calib_Read import Calib_Read
from Atmo_Read import Atmo_Read
from trust import trust
import matplotlib.pyplot as plt

import numpy as np

    # Fichier texte qui contient les bandes spectrales
path_calib = 'TEST_Thunmpy_08-07-2019/Filters_TASI.txt'
    # Fichier texte qui contient le niveau de bruit du capteur
path_noise = 'TEST_Thunmpy_08-07-2019/Noise_TASI.txt'
    # Fichier tape7 qui contient les termes atmosphériques en sortie de
    # MODTRAN
path_tape7 = 'TEST_Thunmpy_08-07-2019/27June-1132_Manu.tp7'
    # Fichier texte qui liste les émissivités prises en compte pour
    # l'établissement de la loi de calibration "emin = f(MMD)"
#path_mmdcalib = 'data/emissivity/list_path.txt';

### Parameters à modifier :
    # Nombre de pixels purs à sélectionner par matériaux
Npix = 5;
    # Nombre de matériaux présent dans la scène
Nmat = 3;
    # Gamma : hyperparamètre pour la méthode TRUST
Gamma = 5*10**-3

### Lecture de l'image
#[Image, HdrFile] = funct_ReadHS(path_image, 'Ext', '.bsq', 'Int', 'bsq');

### Lecture du fichier de calibration (capteur)
Calib = Calib_Read(path_calib, 'Noise', path_noise, 'Print', 0)

### Lecture de l'atmosphère
Atmo = Atmo_Read(path_tape7, 'Calib', Calib)

### Calibration de la méthode TES
#MMD_Law = tes_mmdCalib(path_mmdcalib, Calib);

### Construction de la LUT de la loi du corps noir pour l'instrument TASI
#LcnLUT = Make_LcnLUT(Calib);

### Construction des cartes d'émissivité
#[EmissTES, TempTES] = tes_run(Image, MMD_Law, Atmo, LcnLUT);
    ### Construction des émissivités et des températures en 2D
#E_2D = reshape(EmissTES, [size(Image,1)*size(Image,2),size(Image,3)]);
#T_2D = reshape(TempTES, [size(Image,1)*size(Image,2),1]);

### Localisation des estimation des émissivités et des températures
### moyennes pour chaque matériaux
    ### Représentation de la longueur d'onde 10µm (bande #19)
#figure, Color_Image(Image(:,:,19));
    ### initialisation des vecteurs E_Vect et T_Vect
#E_Vect = zeros(Nmat,size(Image,3)); T_Vect = zeros(Nmat,1);
#for mat = 1:Nmat, 
    ### Construction du titre
#    title({'TIR Image at \lambda=10µm' ...
#        sprintf('Localisation of Material # %d',mat)});
    ### Localisation sur l'image des Npix pixels purs
#    [x,y] = ginput(Npix); x = round(x); y = round(y);
#    Ind = sub2ind([size(Image,1),size(Image,2)],y,x);
    ### Estimation de l'émissivité moyenne pour ce matériau
#    E_Vect(mat,:) = mean(E_2D(Ind(:),:));
    ### Estimation de la température moyenne pour ce matériau
#    T_Vect(mat,1) = mean(T_2D(Ind(:),:));
#end, close(gcf)

# For this test
Im_tmp=np.loadtxt('TEST_Thunmpy_08-07-2019/Image.txt')
Image=np.reshape(Im_tmp,(18,15,32))

E_Vect=np.loadtxt('TEST_Thunmpy_08-07-2019/E_Vect.txt')

T_Vect=np.loadtxt('TEST_Thunmpy_08-07-2019/T_Vect.txt')

### Estimation des abondances avec la méthode TRUST
S,Tsub = trust(Image, E_Vect.T, T_Vect.T, Atmo,'gamma', Gamma,'printf', 'on', 'nargout',0)

### Représentation des pôles de mélanges et des températures subpixéliques
### avec la méthode TRUST

from osgeo import gdal, gdalconst

path_out='TEST_Thunmpy_08-07-2019/Abondance.bsq'

S.shape
cols=len(S[:,0,0])
rows=len(S[0,:,0])
mat=len(S[0,0,:])


driver = gdal.GetDriverByName("ENVI")
outDs = driver.Create(path_out,rows,cols,mat,gdal.GDT_Float32)

for i in range(mat):
	outBand = outDs.GetRasterBand(i+1)
	outBand.WriteArray(S[:,:,i])
	outBand.FlushCache()
	
	
path_out='TEST_Thunmpy_08-07-2019/Temperature.bsq'
	
Tsub.shape
cols=len(Tsub[:,0,0])
rows=len(Tsub[0,:,0])
mat=len(Tsub[0,0,:])

driver = gdal.GetDriverByName("ENVI")
outDs = driver.Create(path_out,rows,cols,mat,gdal.GDT_Float32)
for i in range(mat):
	outBand = outDs.GetRasterBand(i+1)
	outBand.WriteArray(Tsub[:,:,i])
	outBand.FlushCache()
	

    ### Visualisation des abondances
plt.subplots(nrows=1,ncols=3)
for mat in range(0,Nmat):
	plt.subplot(1,3,mat+1)
	plt.pcolormesh(S[:,:,mat])
plt.show()
plt.savefig("Visualisation des abondances.png")
    ### Visualisation des températures subpixéliques
plt.subplots(nrows=1,ncols=3)
for mat in range(0,Nmat):
	plt.subplot(1,3,mat+1) 
	plt.pcolormesh(Tsub[:,:,mat],vmin=290, vmax=330)
plt.show()
plt.savefig("Visualisation des températures subpixéliques.png")

#for mat = 1:Nmat, subplot(2,Nmat,Nmat+mat), 
#    Color_Image(Tsub(:,:,mat),[min(Tsub(:))-1,max(Tsub(:))+1]); end

