'''
################ Needed packages #### 
numpy
gdal
Thunmpy
'''
import numpy as np
from osgeo import gdal
from Thunmpy import ThunmFit
from Thunmpy import Thunmixing
from Thunmpy import Thunmcorr

def TsHARP(T_C, I_C, I_H, iscale, min_T=285, path_image=False):
    
    '''
    This function applies TsHARP method to sharpen T_C with Index I_C and I_H.
    	
    ############## Inputs:
    
    T_C: path to LST image at coarse resolution
    
    I_C: path to Index image at coarse resolution	
    
    I_H: path to Index image at high resolution	
    
    iscale: factor of upscaling. For example from 100m to 20m iscale=5
    
    min_T (optional): Minimum expected temperature in the image. All the temerpature 
           below this one are considered spurious measures and not taken into account in the 
           regression.	
    
    path_image (optional): path with the folder and filename where saving the sharpened LST image.
    
    ############## Outputs:
    
    T_H_corrected : corrected LST at high resolution
    
    #
    ###
    ######
    ###########
    ############### C. Granero-Belinchon; IMT Atlantique, Brest; 02/2022
    '''

    # Reading index at coarse resolution
    filename_index = I_C
    dataset = gdal.Open(filename_index)
    cols = dataset.RasterXSize #Spatial dimension x
    rows = dataset.RasterYSize #Spatial dimension y
    index_c = dataset.ReadAsArray(0, 0, cols, rows).astype(np.float)
            
    # Reading temperature at coarse resolution
    TempFile = T_C
    dataset_Temp = gdal.Open(TempFile)
    cols_t = dataset_Temp.RasterXSize #Spatial dimension x
    rows_t = dataset_Temp.RasterYSize #Spatial dimension y
    Temp_c= dataset_Temp.ReadAsArray(0, 0, cols_t, rows_t).astype(np.float)

    # Reading index at high resolution
    filename_index = I_H
    dataset = gdal.Open(filename_index)
    cols = dataset.RasterXSize #Spatial dimension x
    rows = dataset.RasterYSize #Spatial dimension y
    projection_h = dataset.GetProjection()
    geotransform_h = dataset.GetGeoTransform()
    index_h = dataset.ReadAsArray(0, 0, cols, rows).astype(np.float)


    fit=ThunmFit.linear_fit(index_c, Temp_c, min_T, path_fit=False, path_plot=False)
    T_H= Thunmixing.linear_unmixing(index_h, Temp_c, fit, iscale, mask=0)
    T_H_corrected=Thunmcorr.correction_linreg(index_c, Temp_c, T_H, iscale, fit)
    
    if path_image!=False:
        driver = gdal.GetDriverByName("ENVI")
        outDs = driver.Create(path_image, int(cols), int(rows), 1, gdal.GDT_Float32)
        outDs.SetProjection(projection_h)
        outDs.SetGeoTransform(geotransform_h)
        outBand = outDs.GetRasterBand(1)
        outBand.WriteArray(T_H_corrected, 0, 0)
        outBand.FlushCache()
        outDs = None

    print('TsHARP Done')
    
    return T_H_corrected
    
def HUTS(T_C, I_C, I_H, A_C, A_H, iscale, min_T=285, min_threshold=263, max_threshold=323, p=False, path_image=False):
    
    '''
    This function applies HUTS method to sharpen T_C with Index I_C, A_C and I_H, A_H.
    	
    ############## Inputs:
    
    T_C: path to LST image at coarse resolution
    
    I_C: path to Index image at coarse resolution	
    
    I_H: path to Index image at high resolution	
    
    A_C: path to Albedo (VNIR-SWIR Index) image at coarse resolution	
    
    A_H: path to Albedo (VNIR-SWIR Index) image at high resolution	
    
    iscale: factor of upscaling. For example from 100m to 20m iscale=5
    
    min_T (optional): Minimum expected temperature in the image. All the temerpature 
           below this one are considered spurious measures and not taken into account in the 
           regression.	
           
    max_threshold and min_threshold (optional): maximum and minimum LSTs taken into account in the residual correction step.
    
    p : initial guess for fit parameters
    
    path_image (optional): path with the folder and filename where saving the sharpened LST image.
    
    ############## Outputs:
    
    T_H_corrected : corrected LST at high resolution
    
    #
    ###
    ######
    ###########
    ############### C. Granero-Belinchon; IMT Atlantique, Brest; 02/2022
    '''

    # Reading index at coarse resolution
    filename_index = I_C
    dataset = gdal.Open(filename_index)
    cols = dataset.RasterXSize #Spatial dimension x
    rows = dataset.RasterYSize #Spatial dimension y
    index_c = dataset.ReadAsArray(0, 0, cols, rows).astype(np.float)
            
    # Reading temperature at coarse resolution
    TempFile = T_C
    dataset_Temp = gdal.Open(TempFile)
    cols_t = dataset_Temp.RasterXSize #Spatial dimension x
    rows_t = dataset_Temp.RasterYSize #Spatial dimension y
    Temp_c= dataset_Temp.ReadAsArray(0, 0, cols_t, rows_t).astype(np.float)

    # Reading index at high resolution
    filename_index = I_H
    dataset = gdal.Open(filename_index)
    cols = dataset.RasterXSize #Spatial dimension x
    rows = dataset.RasterYSize #Spatial dimension y
    projection_h = dataset.GetProjection()
    geotransform_h = dataset.GetGeoTransform()
    index_h = dataset.ReadAsArray(0, 0, cols, rows).astype(np.float)

    # Reading albedo at coarse resolution
    filename_albedo = A_C
    dataset_alb = gdal.Open(filename_albedo)
    cols = dataset_alb.RasterXSize #Spatial dimension x
    rows = dataset_alb.RasterYSize #Spatial dimension y
    albedo_c = dataset_alb.ReadAsArray(0, 0, cols, rows).astype(np.float)
    
    # Reading index at high resolution
    filename_albedo = A_H
    dataset_alb = gdal.Open(filename_albedo)
    cols = dataset_alb.RasterXSize #Spatial dimension x
    rows = dataset_alb.RasterYSize #Spatial dimension y
    albedo_h = dataset_alb.ReadAsArray(0, 0, cols, rows).astype(np.float)

    fit=ThunmFit.huts_fit(index_c, albedo_c, Temp_c, min_T, p, path_fit=False)
    T_H= Thunmixing.huts_unmixing(index_h, albedo_h, Temp_c, fit, iscale, mask=0)
    T_H_corrected=Thunmcorr.quality_correction(Temp_c, T_H, iscale, max_threshold=max_threshold, min_threshold=min_threshold)
    
    if path_image!=False:
        driver = gdal.GetDriverByName("ENVI")
        outDs = driver.Create(path_image, int(cols), int(rows), 1, gdal.GDT_Float32)
        outDs.SetProjection(projection_h)
        outDs.SetGeoTransform(geotransform_h)
        outBand = outDs.GetRasterBand(1)
        outBand.WriteArray(T_H_corrected, 0, 0)
        outBand.FlushCache()
        outDs = None

    print('HUTS Done')
    
    return T_H_corrected

def ATPRK(T_C, I_C, I_H, iscale, scc, block_size=5, sill=7, ran=1000, min_T=285, path_image=False):
    
    '''
    This function applies ATPRK method to sharpen T_C with Index I_C and I_H.
    	
    ############## Inputs:
    
    T_C: path to LST image at coarse resolution
    
    I_C: path to Index image at coarse resolution	
    
    I_H: path to Index image at high resolution	

    iscale: factor of upscaling. For example from 100m to 20m iscale=5
    
    scc: The coarse pixel has a size scc * scc
    
    block_size: The moving window used to compute the semivariograms and the weights in the 
                ATPK procedures has a size block_size * block_size 
    
    sill and ran are the parameters defining the shape of the semivariogram.
     
    min_T (optional): Minimum expected temperature in the image. All the temerpature 
           below this one are considered spurious measures and not taken into account in the 
           regression.	
           
    path_image (optional): path with the folder and filename where saving the sharpened LST image.
    
    ############## Outputs:
    
    T_H_corrected : corrected LST at high resolution
    
    #
    ###
    ######
    ###########
    ############### C. Granero-Belinchon; IMT Atlantique, Brest and A. Michel; ONERA-DOTA, Toulouse; 02/2022
    '''

    # Reading index at coarse resolution
    filename_index = I_C
    dataset = gdal.Open(filename_index)
    cols = dataset.RasterXSize #Spatial dimension x
    rows = dataset.RasterYSize #Spatial dimension y
    index_c = dataset.ReadAsArray(0, 0, cols, rows).astype(np.float)
            
    # Reading temperature at coarse resolution
    TempFile = T_C
    dataset_Temp = gdal.Open(TempFile)
    cols_t = dataset_Temp.RasterXSize #Spatial dimension x
    rows_t = dataset_Temp.RasterYSize #Spatial dimension y
    Temp_c= dataset_Temp.ReadAsArray(0, 0, cols_t, rows_t).astype(np.float)

    # Reading index at high resolution
    filename_index = I_H
    dataset = gdal.Open(filename_index)
    cols = dataset.RasterXSize #Spatial dimension x
    rows = dataset.RasterYSize #Spatial dimension y
    projection_h = dataset.GetProjection()
    geotransform_h = dataset.GetGeoTransform()
    index_h = dataset.ReadAsArray(0, 0, cols, rows).astype(np.float)

    fit=ThunmFit.linear_fit(index_c, Temp_c, min_T, path_fit=False, path_plot=False)
    T_H= Thunmixing.linear_unmixing(index_h, Temp_c, fit, iscale, mask=0)
    T_H_corrected=Thunmcorr.correction_ATPRK(index_c, Temp_c, T_H, fit, iscale, scc, block_size, sill, ran, path_plot=False)
    
    if path_image!=False:
        driver = gdal.GetDriverByName("ENVI")
        outDs = driver.Create(path_image, int(cols), int(rows), 1, gdal.GDT_Float32)
        outDs.SetProjection(projection_h)
        outDs.SetGeoTransform(geotransform_h)
        outBand = outDs.GetRasterBand(1)
        outBand.WriteArray(T_H_corrected, 0, 0)
        outBand.FlushCache()
        outDs = None

    print('ATPRK Done')
    
    return T_H_corrected

def AATPRK(T_C, I_C, I_H, iscale, scc, b_radius=2, block_size=5, sill=7, ran=1000, min_T=285, path_image=False):
    
    '''
    This function applies AATPRK method to sharpen T_C with Index I_C and I_H.
    	
    ############## Inputs:
    
    T_C: path to LST image at coarse resolution
    
    I_C: path to Index image at coarse resolution	
    
    I_H: path to Index image at high resolution	

    iscale: factor of upscaling. For example from 100m to 20m iscale=5
    
    scc: The coarse pixel has a size scc * scc
    
    block_size: The moving window used to compute the semivariograms and the weights in the 
                ATPK procedures has a size block_size * block_size 
    
    sill and ran are the parameters defining the shape of the semivariogram.
     
    min_T (optional): Minimum expected temperature in the image. All the temerpature 
           below this one are considered spurious measures and not taken into account in the 
           regression.	
           
    path_image (optional): path with the folder and filename where saving the sharpened LST image.
    
    ############## Outputs:
    
    T_H_corrected : corrected LST at high resolution
    
    #
    ###
    ######
    ###########
    ############### C. Granero-Belinchon; IMT Atlantique, Brest and A. Michel; ONERA-DOTA, Toulouse; 02/2022
    '''

    # Reading index at coarse resolution
    filename_index = I_C
    dataset = gdal.Open(filename_index)
    cols = dataset.RasterXSize #Spatial dimension x
    rows = dataset.RasterYSize #Spatial dimension y
    index_c = dataset.ReadAsArray(0, 0, cols, rows).astype(np.float)
            
    # Reading temperature at coarse resolution
    TempFile = T_C
    dataset_Temp = gdal.Open(TempFile)
    cols_t = dataset_Temp.RasterXSize #Spatial dimension x
    rows_t = dataset_Temp.RasterYSize #Spatial dimension y
    Temp_c= dataset_Temp.ReadAsArray(0, 0, cols_t, rows_t).astype(np.float)

    # Reading index at high resolution
    filename_index = I_H
    dataset = gdal.Open(filename_index)
    cols = dataset.RasterXSize #Spatial dimension x
    rows = dataset.RasterYSize #Spatial dimension y
    projection_h = dataset.GetProjection()
    geotransform_h = dataset.GetGeoTransform()
    index_h = dataset.ReadAsArray(0, 0, cols, rows).astype(np.float)

    Intercept, Slope =ThunmFit.linear_fit_window(index_c, Temp_c, min_T, b_radius)
    T_H= Thunmixing.aatprk_unmixing(index_h, Temp_c, np.asarray([Slope,Intercept]), iscale)
    T_H_corrected=Thunmcorr.correction_AATPRK(index_c, Temp_c, T_H, np.asarray([Slope,Intercept]), iscale, scc, block_size, sill, ran, path_plot=False)
    
    if path_image!=False:
        driver = gdal.GetDriverByName("ENVI")
        outDs = driver.Create(path_image, int(cols), int(rows), 1, gdal.GDT_Float32)
        outDs.SetProjection(projection_h)
        outDs.SetGeoTransform(geotransform_h)
        outBand = outDs.GetRasterBand(1)
        outBand.WriteArray(T_H_corrected, 0, 0)
        outBand.FlushCache()
        outDs = None

    print('AATPRK Done')
    
    return T_H_corrected
