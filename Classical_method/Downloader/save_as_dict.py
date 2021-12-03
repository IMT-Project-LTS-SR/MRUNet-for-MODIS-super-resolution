import os 
import numpy as np
root_dir = '/homes/g19tian/Documents/1_Projet3A/MODIS/'
LST_1km_dir = os.path.join(root_dir,"MOD_2017_MOD11A1/tifs_files/1km")
LST_2km_dir = os.path.join(root_dir,"MOD_2017_MOD11A1/tifs_files/2km")
NDVI_1km_dir = os.path.join(root_dir,"MOD_2017_MOD13A2/tifs_files/1km")
NDVI_2km_dir = os.path.join(root_dir,"MOD_2017_MOD13A2/tifs_files/2km")
LST_1km_files = os.listdir(LST_1km_dir)
LST_2km_files = os.listdir(LST_2km_dir)
NDVI_1km_files = os.listdir(NDVI_1km_dir)
NDVI_2km_files = os.listdir(NDVI_2km_dir)

LST_to_NDVI_dic_path = os.path.join(root_dir,"LST_to_NDVI_dic_2017.npy")
# Generate the LST to NDVI one-to-one dictionary
LST_to_NDVI_dic = {}
No_NDVI = []
for idx in range(len(LST_1km_files)):
	name_list = LST_1km_files[idx].split(".")
	date = int(name_list[1].strip("A")[-3:])
	patch_num = name_list[-2]
	ndvi_date = "MOD13A2.A2017"+ str(date//16 * 16 + 1).zfill(3)
	for ndvi_file in NDVI_1km_files:
		if ndvi_date in ndvi_file and ndvi_file.endswith(patch_num+".tif"):
			ndvi_1km_path = os.path.join(NDVI_1km_dir, ndvi_date)
			ndvi_2km_path = os.path.join(NDVI_2km_dir, ndvi_date)
			# ndvi_1km, ndvi_2km = get_ndvi(ndvi_1km_path, ndvi_2km_path)
			# print("LST: {} and ndvi: {} and ndvi_date: {}".format(LST_1km_files[idx], ndvi_file,ndvi_date))
			LST_to_NDVI_dic[LST_1km_files[idx]] = ndvi_file 
			break
	if not LST_1km_files[idx] in LST_to_NDVI_dic.keys():
		No_NDVI.append(LST_1km_files[idx])
		os.remove(os.path.join(LST_1km_dir, LST_1km_files[idx]))
		os.remove(os.path.join(LST_2km_dir, LST_1km_files[idx]))




# for idx in range(len(LST_1km_files)):
# 	if not LST_1km_files[idx] in LST_to_NDVI_dic.keys():
de = 0

np.save(LST_to_NDVI_dic_path, LST_to_NDVI_dic)