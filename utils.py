import os
from pymodis import downmodis
import matplotlib.pyplot as plt
import numpy as np
from osgeo import gdal
import skimage.measure
	
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
