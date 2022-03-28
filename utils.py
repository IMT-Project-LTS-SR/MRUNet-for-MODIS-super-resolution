import os
from pymodis import downmodis
import matplotlib.pyplot as plt
import numpy as np
from osgeo import gdal
import skimage.measure
import cv2
from skimage.metrics import peak_signal_noise_ratio
from skimage.metrics import structural_similarity as ssim_sk
	
def norm4_f2(a,axis):
	# result = np.zeros((np.array(a.shape)/2).astype(int))
	result = np.zeros([a.shape[0],a.shape[1]])
	for i in range(a.shape[0]):
		for j in range(a.shape[1]):
			non_0 = np.sum(a[i,j,:,:]!=0)
			if non_0 != 4:
				result[i,j] = 0
			else:
				result[i,j] = ((np.sum(a[i,j,:,:]**4))/non_0)**0.25

	return result

def norm4_f4(a,axis):
	# result = np.zeros((np.array(a.shape)/2).astype(int))
	result = np.zeros([a.shape[0],a.shape[1]])
	for i in range(a.shape[0]):
		for j in range(a.shape[1]):
			non_0 = np.sum(a[i,j,:,:]!=0)
			if non_0 != 16:
				result[i,j] = 0
			else:
				result[i,j] = ((np.sum(a[i,j,:,:]**4))/non_0)**0.25

	return result


def sliding_window(img, step, window_size):
	"""
	INPUT:
	img = input image
	step = the distance (in pixels, for both horizontal and vertical direction) 
	between 2 windows
	window_size = size of the sliding window, format: (pixels(y-axis),pixels(x-axis))

	OUTPUT: (x,y,window)
	x,y: the coordinates of the top left corner of the current window (in pixels) 
	in the coordinate system of "img".
	window: the current window, size = (window_size)
	"""
	for y in range(0, img.shape[0], step):
		for x in range(0, img.shape[1], step):
			yield (x, y, img[y:y + window_size[1], x:x + window_size[0]])

def read_tif(in_file):
	"""
	Similar to read_modis but this function is for reading output .tif file from
	"""
	# save_tif()

	dataset =	gdal.Open(in_file, gdal.GA_ReadOnly)

	cols =dataset.RasterXSize
	rows = dataset.RasterYSize
	projection = dataset.GetProjection()
	geotransform = dataset.GetGeoTransform()

	# Coordinates of top left pixel of the image (Lat, Lon)
	coords=np.asarray((geotransform[0],geotransform[3]))

	# Open dataset Day as np.array()
	band = dataset.GetRasterBand(1)
	LST_K_day = band.ReadAsArray().astype(np.float)
	bandtype = gdal.GetDataTypeName(band.DataType)

	# open dataset Night as np.array()
	band = dataset.GetRasterBand(2)
	LST_K_night = band.ReadAsArray().astype(np.float)
	bandtype = gdal.GetDataTypeName(band.DataType)
	dataset = None

	return LST_K_day,LST_K_night,cols,rows,projection,geotransform

def save_tif(out_file, LST_K_day, LST_K_night, cols, rows, projection, geotransform):
	"""
	INPUT:
	out_file: directory + name of output tif file (e.g a/b/c/MOD11A1.tif)
	LST_K_day, LST_K_night, cols, rows, projection, geotransform: read read_modis

	OUTPUT:
	A .tif file having 2 bands(1 for day, 1 for night)
	"""

	# Eliminate the clouds' pixel
	num_px = LST_K_day.shape[0]*LST_K_day.shape[1]
	# thres = 0.15 # Threshold: maximum number of sea/cloud pixels
	thres = 0 # Threshold: maximum number of sea/cloud pixels
	
	if len(LST_K_day[LST_K_day==0.0]) > thres*num_px or len(LST_K_night[LST_K_night==0.0]) > thres*num_px:
		return False

	bands = 2
	day_night_imgs = np.zeros((2,LST_K_day.shape[0],LST_K_day.shape[1]))
	day_night_imgs[0,:,:]= LST_K_day
	day_night_imgs[1,:,:]= LST_K_night

	driver = gdal.GetDriverByName("GTiff")
	outDs = driver.Create(out_file, day_night_imgs.shape[1], day_night_imgs.shape[2], bands, gdal.GDT_Float32) 
	outDs.SetProjection(projection)
	outDs.SetGeoTransform(geotransform) 

	for i in range(1,bands+1):
		outBand = outDs.GetRasterBand(i)
		outBand.WriteArray(day_night_imgs[i-1,:,:])
		outDs.FlushCache()
		
	return True

def read_modis(in_file):
	"""
	This function is for reading modis file in format "x.hdf" contained in 
	in_file parameter.
	INPUT: 
	in_file: path to hdf file
	to be completed if we need more stuffs

	OUTPUT:
	LST_K_day and LST_K_night: Day and Night LST image arrays
	cols, rows: numbers of pixels alongs x,y axes of each raster
	projection: projection information
	geotransform: georeferencing information
	# georef format = (x-coor top left pixel,resX,0,y-coor top left pixel,rexY,0)
	to be completed if we need more information (georeference,bands,etc.)
	"""
	# open dataset Day
	dataset = gdal.Open(in_file,gdal.GA_ReadOnly)
	subdataset =	gdal.Open(dataset.GetSubDatasets()[0][0], gdal.GA_ReadOnly)

	cols =subdataset.RasterXSize
	rows = subdataset.RasterYSize
	projection = subdataset.GetProjection()
	geotransform = subdataset.GetGeoTransform()

	# Coordinates of top left pixel of the image (Lat, Lon)
	# coords=np.asarray((geotransform[0],geotransform[3]))

	# We read the Image as an array
	band = subdataset.GetRasterBand(1)
	LST_raw = band.ReadAsArray(0, 0, cols, rows).astype(np.float)
	# bandtype = gdal.GetDataTypeName(band.DataType)

	# To convert LST MODIS units to Kelvin
	LST_K_day=0.02*LST_raw

	dataset = None
	subdataset = None
	# open dataset Night
	dataset = gdal.Open(in_file,gdal.GA_ReadOnly)
	subdataset =	gdal.Open(dataset.GetSubDatasets()[4][0], gdal.GA_ReadOnly)

	# We read the Image as an array
	band = subdataset.GetRasterBand(1)
	LST_raw = band.ReadAsArray(0, 0, cols, rows).astype(np.float)
	# bandtype = gdal.GetDataTypeName(band.DataType)

	# To convert LST MODIS units to Kelvin
	LST_K_night=0.02*LST_raw
	dataset = None
	subdataset = None

	# return LST_K_day, LST_K_night
	return LST_K_day, LST_K_night, cols, rows, projection, geotransform


def crop_modis(hdf_path, hdf_name, save_dir, save_dir_downsample_2, save_dir_downsample_4,step=64,size=(64,64)):
	"""
	INPUT:
	hdf_path = input image path to be cropped | or hdf file path ("/a/b/c.hdf")
	save_dir = directory for saving cropped images
	step, size: parameters of "sliding_window()"

	OUTPUT: images cropped from the image in hdf_path, saved to save_dir
	"""
	if not hdf_path.endswith('hdf'): 
		print("Not hdf file Sorry!")
		return 

	img_day, img_night, cols, rows, projection, geotransform = read_modis(hdf_path)
	
	img_days = []
	img_nights = []
	img_cropped_names = []
	geotransform2s = []
	cols2, rows2 = size

	if img_day is None or img_night is None:
		print("Cannot handle this MODIS file: ", hdf_path, ". Please check it again")
		return
	# For day image
	win_count = 0
	for (x,y,window) in sliding_window(img_day, step, size):
			if window.shape[0] != size[0] or window.shape[1] != size[1]:
					continue

			img_cropped_name = hdf_name + ".{}.tif".format(str(win_count).zfill(4))
			img_cropped = window
			geotransform2 = np.asarray(geotransform)
			geotransform2[0] = geotransform[0]+x*geotransform[1] # 1st coordinate of top left pixel of the image 
			geotransform2[3] = geotransform[3]+y*geotransform[5] # 2nd coordinate of top left pixel of the image
			geotransform2=tuple(geotransform2)

			img_cropped_names.append(img_cropped_name)
			img_days.append(img_cropped)
			geotransform2s.append(geotransform2)
			
			win_count += 1
	# print("Number of cropped day images", win_count)

	# For night image
	win_count = 0
	for (x,y,window) in sliding_window(img_night, step, size):
		if window.shape[0] != size[0] or window.shape[1] != size[1]:
				continue
		# save_path = os.path.join(save_dir,img_cropped_name)
		img_cropped = window
		# np.save(save_path,img_cropped)
		img_nights.append(img_cropped)
		win_count += 1
	# print("Number of cropped night images", win_count)
	
	# Save images and metadata into .tif file
	for i in range(len(img_cropped_names)):
		save_path = os.path.join(save_dir,img_cropped_names[i])
		succes = save_tif(save_path, img_days[i], img_nights[i], cols2, rows2, projection, geotransform2s[i])

		if succes:
			save_path_downsample_2 = os.path.join(save_dir_downsample_2,img_cropped_names[i])
			save_path_downsample_4 = os.path.join(save_dir_downsample_4,img_cropped_names[i])

			img_day_downsample_2 = skimage.measure.block_reduce(img_days[i],(2,2),norm4_f2)
			img_night_downsample_2 = skimage.measure.block_reduce(img_nights[i],(2,2),norm4_f2)
			img_day_downsample_4 = skimage.measure.block_reduce(img_days[i],(4,4),norm4_f4)
			img_night_downsample_4 = skimage.measure.block_reduce(img_nights[i],(4,4),norm4_f4)

			_ = save_tif(save_path_downsample_2, img_day_downsample_2, img_night_downsample_2, cols2, rows2, projection, geotransform2s[i])
			_ = save_tif(save_path_downsample_4, img_day_downsample_4, img_night_downsample_4, cols2, rows2, projection, geotransform2s[i])

			# print("img_night_downsample",img_night_downsample)
			# # Display image
			# tif_1km_path = save_path
			# tif_2km_path = save_path_downsample
			# LST_K_day_1km, LST_K_night_1km, cols_1km, rows_1km, projection_1km, geotransform_1km = read_tif(tif_1km_path)
			# LST_K_day_2km, LST_K_night_2km, cols_2km, rows_2km, projection_2km, geotransform_2km = read_tif(tif_2km_path)

			# plt.figure()
			# plt.subplot(121)
			# plt.imshow(LST_K_day_1km)
			# plt.clim(260,301)
			# plt.colorbar()
			# plt.title("1km")
			# plt.subplot(122)
			# plt.imshow(LST_K_day_2km)
			# plt.clim(260,301)
			# plt.colorbar()
			# plt.title("2km")
			# plt.show()
		else:
			# print("Not success!")
			pass


def save_tif_MOD13A2(out_file, red_downsample, NIR_downsample, MIR_downsample, cols, rows, projection, geotransform):
# def save_tif(out_file, LST_K_day, LST_K_night, cols, rows, projection, geotransform):

	# Eliminate the clouds' pixel
	num_px = red_downsample.shape[0]*red_downsample.shape[1]
	thres = 0 # Threshold: maximum number of sea/cloud pixels
	# thres = 0.15 # Threshold: maximum number of sea/cloud pixels
	if len(red_downsample[red_downsample==0.0]) > thres*num_px or len(NIR_downsample[NIR_downsample==0.0]) > thres*num_px or len(MIR_downsample[MIR_downsample==0.0]) > thres*num_px:
		return False

	bands = 3
	red_NIR_MIR_imgs = np.zeros((3,red_downsample.shape[0],red_downsample.shape[1]))
	red_NIR_MIR_imgs[0,:,:]= red_downsample
	red_NIR_MIR_imgs[1,:,:]= NIR_downsample
	red_NIR_MIR_imgs[2,:,:]= MIR_downsample

	driver = gdal.GetDriverByName("GTiff")
	outDs = driver.Create(out_file, red_NIR_MIR_imgs.shape[1], red_NIR_MIR_imgs.shape[2], bands, gdal.GDT_Float32) 
	outDs.SetProjection(projection)
	outDs.SetGeoTransform(geotransform) 

	for i in range(1,bands+1):
		outBand = outDs.GetRasterBand(i)
		outBand.WriteArray(red_NIR_MIR_imgs[i-1,:,:])
		outDs.FlushCache()
		
	return True

def read_modis_MOD13A2(in_file):
	# in_file = "MODIS/MOD_2020/hdfs_files/MOD13A2.A2020001.h18v04.061.2020326033640.hdf"
	# in_file = '/content/drive/MyDrive/Ganglin/1_IMT/MCE_Projet3A/MODIS/MOD_2020/hdfs_files/MOD11A1.A2020001.h18v04.061.2021003092415.hdf'
	dataset = gdal.Open(in_file,gdal.GA_ReadOnly)
	subdataset =	gdal.Open(dataset.GetSubDatasets()[3][0], gdal.GA_ReadOnly)

	cols =subdataset.RasterXSize
	rows = subdataset.RasterYSize
	projection = subdataset.GetProjection()
	geotransform = subdataset.GetGeoTransform()

	# We read the Image as an array
	band = subdataset.GetRasterBand(1)
	LST_raw = band.ReadAsArray(0, 0, cols, rows).astype(np.float)

	# To convert LST MODIS units to Kelvin
	red =0.02*LST_raw

	# open dataset Night
	dataset = gdal.Open(in_file,gdal.GA_ReadOnly)
	subdataset =	gdal.Open(dataset.GetSubDatasets()[4][0], gdal.GA_ReadOnly)

	# We read the Image as an array
	band = subdataset.GetRasterBand(1)
	LST_raw = band.ReadAsArray(0, 0, cols, rows).astype(np.float)
	# bandtype = gdal.GetDataTypeName(band.DataType)
	# To convert LST MODIS units to Kelvin
	NIR =0.02*LST_raw

	# open dataset Night
	dataset = gdal.Open(in_file,gdal.GA_ReadOnly)
	subdataset =	gdal.Open(dataset.GetSubDatasets()[6][0], gdal.GA_ReadOnly)

	# We read the Image as an array
	band = subdataset.GetRasterBand(1)
	LST_raw = band.ReadAsArray(0, 0, cols, rows).astype(np.float)
	# bandtype = gdal.GetDataTypeName(band.DataType)
	# To convert LST MODIS units to Kelvin
	MIR =0.02*LST_raw
	return red, NIR, MIR, cols, rows, projection, geotransform

def crop_modis_MOD13A2(hdf_path, hdf_name, save_dir, save_dir_downsample_2, save_dir_downsample_4,step=64,size=(64,64)):
	"""
	INPUT:
	hdf_path = input image path to be cropped | or hdf file path ("/a/b/c.hdf")
	save_dir = directory for saving cropped images
	step, size: parameters of "sliding_window()"

	OUTPUT: images cropped from the image in hdf_path, saved to save_dir
	"""
	if not hdf_path.endswith('hdf'): 
		print("Not hdf file Sorry!")
		return 

	img_day, img_night, cols, rows, projection, geotransform = read_modis(hdf_path)
	
	img_days = []
	img_nights = []
	img_cropped_names = []
	geotransform2s = []
	cols2, rows2 = size

	if img_day is None or img_night is None:
		print("Cannot handle this MODIS file: ", hdf_path, ". Please check it again")
		return

	red, NIR, MIR, cols, rows, projection, geotransform = read_modis_MOD13A2(hdf_path)

	reds = []
	NIRs = []
	MIRs = []
	img_cropped_names = []
	geotransform2s = []
	cols2, rows2 = size

	if red is None or NIR is None or MIR is None:
		print("Cannot handle this MODIS file: ", hdf_path, ". Please check it again")
	# For day image
	win_count = 0
	for (x,y,window) in sliding_window(red, step, size):
		if window.shape[0] != size[0] or window.shape[1] != size[1]:
				continue

		img_cropped_name = hdf_name + ".{}.tif".format(str(win_count).zfill(4))
		img_cropped = window
		geotransform2 = np.asarray(geotransform)
		geotransform2[0] = geotransform[0]+x*geotransform[1] # 1st coordinate of top left pixel of the image 
		geotransform2[3] = geotransform[3]+y*geotransform[5] # 2nd coordinate of top left pixel of the image
		geotransform2=tuple(geotransform2)

		img_cropped_names.append(img_cropped_name)
		reds.append(img_cropped)
		geotransform2s.append(geotransform2)
		
		win_count += 1
	# print("Number of cropped day images", win_count)
	
	# For NIR image
	win_count = 0
	for (x,y,window) in sliding_window(NIR, step, size):
		if window.shape[0] != size[0] or window.shape[1] != size[1]:
				continue
		img_cropped = window
		# np.save(save_path,img_cropped)
		NIRs.append(img_cropped)
		win_count += 1

	# For MIR image
	win_count = 0
	for (x,y,window) in sliding_window(MIR, step, size):
		if window.shape[0] != size[0] or window.shape[1] != size[1]:
				continue
		img_cropped = window
		# np.save(save_path,img_cropped)
		MIRs.append(img_cropped)
		win_count += 1

	# Save images and metadata into .tif file
	for i in range(len(img_cropped_names)):
		save_path = os.path.join(save_dir,img_cropped_names[i])
		succes = save_tif_MOD13A2(save_path, reds[i], NIRs[i], MIRs[i], cols2, rows2, projection, geotransform2s[i])

		if succes:
			save_path_downsample_2 = os.path.join(save_dir_downsample_2,img_cropped_names[i])
			save_path_downsample_4 = os.path.join(save_dir_downsample_4,img_cropped_names[i])
			red_downsample_2 = skimage.measure.block_reduce(reds[i],(2,2),norm4_f2)
			NIR_downsample_2 = skimage.measure.block_reduce(NIRs[i],(2,2),norm4_f2)
			MIR_downsample_2 = skimage.measure.block_reduce(MIRs[i],(2,2),norm4_f2)

			red_downsample_4 = skimage.measure.block_reduce(reds[i],(4,4),norm4_f4)
			NIR_downsample_4 = skimage.measure.block_reduce(NIRs[i],(4,4),norm4_f4)
			MIR_downsample_4 = skimage.measure.block_reduce(MIRs[i],(4,4),norm4_f4)


			_ = save_tif_MOD13A2(save_path_downsample_2, red_downsample_2, NIR_downsample_2, MIR_downsample_2, cols2, rows2, projection, geotransform2s[i])
			_ = save_tif_MOD13A2(save_path_downsample_4, red_downsample_4, NIR_downsample_4, MIR_downsample_4, cols2, rows2, projection, geotransform2s[i])
			
			
def downsampling(img, scale):
  down_img = np.zeros((int(img.shape[0]/scale), int(img.shape[1]/scale)))
  for y in range(0, img.shape[0], scale):
    for x in range(0, img.shape[1], scale):
      window = img[y:y+scale, x:x+scale].reshape(-1)
      if 0 not in window:
        sum4 = np.sum(window**4)
        norm4 = (sum4/(scale**2))**0.25
      else:
        norm4 = 0.0
      down_img[int(y/scale), int(x/scale)] = norm4
  
  return down_img


def upsampling(img, scale):
  up_img = cv2.resize(img, (img.shape[0]*scale, img.shape[1]*scale), cv2.INTER_CUBIC)
  return up_img


def normalization(image, max_val):
  nml_image = image/max_val
  return nml_image
  

def psnr(label, outputs, max_val):
    """
    Compute Peak Signal to Noise Ratio (the higher the better).
    PSNR = 20 * log10(MAXp) - 10 * log10(MSE).
    https://en.wikipedia.org/wiki/Peak_signal-to-noise_ratio#Definition
    First we need to convert torch tensors to NumPy operable.
    """
    label = label.cpu().detach().numpy()
    outputs = outputs.cpu().detach().numpy()*max_val
    psnr_batch = []
    for i in range(label.shape[0]):
      psnr_img = peak_signal_noise_ratio(label[i,0,:,:], outputs[i,0,:,:], data_range=label.max()-label.min())
      psnr_batch.append(psnr_img)
    return np.array(psnr_batch).mean()

def psnr_notorch(label, outputs):
    """
    Compute Peak Signal to Noise Ratio (the higher the better).
    PSNR = 20 * log10(MAXp) - 10 * log10(MSE).
    https://en.wikipedia.org/wiki/Peak_signal-to-noise_ratio#Definition
    First we need to convert torch tensors to NumPy operable.
    """
    # label = label.cpu().detach().numpy()
    # outputs = outputs.cpu().detach().numpy()
    img_diff = outputs - label
    rmse = math.sqrt(np.mean((img_diff) ** 2))
    if rmse == 0 or rmse==np.inf:
        return 0, np.inf
    else:
        PSNR = 20 * np.log10((np.max(label) - np.min(label)) / rmse)
        return PSNR, rmse


def ssim(label, outputs, max_val):
    label = label.cpu().detach().numpy()
    outputs = outputs.cpu().detach().numpy()*max_val
    ssim_batch = []
    for i in range(label.shape[0]):
      ssim_img = ssim_sk(label[i,0,:,:], outputs[i,0,:,:], data_range=label.max()-label.min())
      ssim_batch.append(ssim_img)
    return np.array(ssim_batch).mean()


def ssim_notorch(label, outputs):
    ssim_map = ssim_sk(label, outputs, data_range=label.max() - label.min())
    return ssim_map



def get_loss(disp, img):
    # THE FINAL LOSS FUNCTION
    mse_img = ((disp - img)**2).mean()
    return mse_img

def linear_fit_test(path_index, path_temperature, min_T, path_fit, plot, path_plot):
    from osgeo import gdal
    import numpy as np
    from scipy.stats import linregress
    import matplotlib.pyplot as plt
################### INDEX ##########################################
    
# Reading index data

    cols = path_index.shape[0] #Spatial dimension x
    rows = path_index.shape[1] #Spatial dimension y

# Read as array the index
    index = path_index

################### Temperature ######################################

# Reading temperature data

    cols_t = path_temperature.shape[0] #Spatial dimension x
    rows_t = path_temperature.shape[1] #Spatial dimension y

# Read as array
    Temp= path_temperature


########### PROCESSED VERSION OF T AND I AND LINEAR REGRESSION #######

    T=Temp.flatten()

    T_great_mask=np.greater(T,0) # We eliminate the background pixels and those with a temperature under min_T
    T=T[T_great_mask]

    I=index.flatten() 
    I=I[T_great_mask]
    
    # We look for nans in the index and we eliminate those pixels from the index and the temperature
    # NaN=np.isfinite(I)
    # I=I[NaN]
    # T=T[NaN]
    
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
        
        # plt.figure(1)
        # plt.scatter(I,T,s=1,c='blue')
        # plt.plot(I,fit_fn(I),'--k')
        # plt.xlabel('I')
        # plt.ylabel('T')
        # plt.grid()
        # plt.savefig(path_plot)
        # plt.close()
        
        print('Fit Done')
        
    else:
        pass
    #     print('Fit Done')
    # print("index",index.shape)
    # print("Temp",Temp.shape)
    return



def linear_unmixing_test(path_index, path_temperature, path_fit, iscale, path_mask, path_out):
    
    import numpy as np
    from osgeo import gdal, gdalconst
    
####################################################################
################ LOADING DATA ######################################
####################################################################

################### INDEX ########################################## # fine scale
    cols = path_index.shape[1] #Spatial dimension x
    rows = path_index.shape[0] #Spatial dimension y
# Read as array the index
    index = path_index

################### Temperature ######################################

# Reading temperature data

    cols_t = path_temperature.shape[1] #Spatial dimension x
    rows_t = path_temperature.shape[0] #Spatial dimension y

# Read as array
    Temp= path_temperature

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
    #    for irow in range(rows):
    #        if I_mask[irow,icol] == False:
    #            T_unm[irow,icol]=0
        
###################################################################
########### SAVING RESULTS ########################################
###################################################################
    
    # print('Unmixing Done')
    # print("index",index.shape)
    # print("Temp",Temp.shape)
    return T_unm


import numpy as np
import matplotlib.pyplot as plt
from osgeo import gdal, gdalconst
import math
from scipy.spatial.distance import pdist, squareform
import scipy.optimize as opt    

def correction_ATPRK_test(path_index, path_fit, path_temperature, path_unm, iscale, scc, block_size, sill, ran, path_out, path_out1, path_out2, path_out3,path_plot,T_unm):
    

####################################################################
################ LOADING DATA ######################################
####################################################################

    b_radius = int(np.floor(block_size/2))
    
################### INDEX Coarse ###################################

    cols_i = path_index.shape[1] #Spatial dimension x
    rows_i = path_index.shape[0] #Spatial dimension y
# Read as array the index
    index = path_index

    cols_t = path_temperature.shape[1] #Spatial dimension x
    rows_t = path_temperature.shape[0] #Spatial dimension y
    Temp= path_temperature


    Tm=Temp
#     TempFile = '/content/drive/MyDrive/Ganglin/1_IMT/MCE_Projet3A/MODIS/MOD_2020/tifs_files/2km_of_MOD11A1.A2020001.h18v04.061.2021003092415.hdf.0015.tif'
#     dataset_Temp = gdal.Open(TempFile)

#     dtset=dataset_Temp
    
# ################### UNMIXED Temperature ###########################        

# #dataset
#     T_unmix = gdal.Open(path_unm)

    cols = T_unm.shape[1] #Spatial dimension x
    rows = T_unm.shape[0] #Spatial dimension y

# Read as array
    TT_unm = T_unm
        
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
    # plt.plot(distances,param_c[0] * (1 - np.exp(-distances/(param_c[1]/3))))
    # plt.plot(distances,Gamma_coarse)
    # plt.savefig(path_plot)
    # plt.close()
    
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
    
    mask_TT_unm=np.zeros((rows,cols))
    indice_mask_TT=np.nonzero(TT_unm)
    mask_TT_unm[indice_mask_TT]=1
    
    # We obtain the delta at the scale of the unmixed temperature
    Delta_T_final =np.zeros((rows,cols))
    # for ic in range(b_radius,math.floor(cols/iscale)-b_radius):
    #     for ir in range(b_radius,math.floor(rows/iscale)-b_radius):
    #         for ir_2 in range((ir-b_radius)*iscale,(ir+b_radius)*iscale+iscale):
    #             for ic_2 in range((ic-b_radius)*iscale-2,(ic+b_radius)*iscale+iscale):
    #                 if mask_TT_unm[ir_2,ic_2]==0:
    #                     Delta_T_final[ir_2,ic_2] = 0
    #                 else:
    #                     temp_var = np.reshape(Delta_T[ir-b_radius:ir+b_radius+1,ic-b_radius:ic+b_radius+1],block_size**2) # We take the block of coarse pixels
    #                     # We multiply each coarse pixel in the block by the corresponding lambda and sum
    #                     Delta_T_final[ir_2,ic_2] = np.sum(lambdas[((ir_2-ir*iscale)*iscale+ic_2-ic*iscale)%lambdas.shape[0],:]*temp_var)
    
    
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

    # plt.figure()
    # plt.imshow(Temp)
    # plt.colorbar()
    # plt.savefig("tmp_fig/Temp.png")

    # plt.figure()
    # plt.imshow(TT_unm)
    # plt.colorbar()
    # plt.savefig("tmp_fig/TT_unm.png")

    # plt.figure()
    # plt.imshow(Delta_T_final)
    # plt.colorbar()
    # plt.savefig("tmp_fig/Delta_T_final.png")

    # plt.figure()
    # plt.imshow(TT_unmixed_corrected)
    # plt.colorbar()
    # plt.savefig("tmp_fig/TT_unmixed_corrected.png")
    # print("index",index.shape)
    # print("Temp",Temp.shape)
    return TT_unmixed_corrected, Delta_T_final, TT_unm

def setup_seed(seed):
    import random
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True

