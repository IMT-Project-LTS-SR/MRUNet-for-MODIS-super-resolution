import os 
import numpy as np
import shutil
LST_to_NDVI_dic = np.load("LST_to_NDVI_dic.npy",allow_pickle=1).item()

ndvi_list = []

lst_path = "MOD_2020_MOD11A1/tifs_files/1km"
lst_name = os.listdir(lst_path)

for name in lst_name:   
    ndvi_list.append(LST_to_NDVI_dic[name])

ndvi_path = "MOD_2020_MOD13A2/tifs_files/1km"
ndvi_name = os.listdir(ndvi_path)

for ndvi in ndvi_name:
    if ndvi not in ndvi_list:
        os.remove(os.path.join("MOD_2020_MOD13A2/tifs_files/1km",ndvi))
        os.remove(os.path.join("MOD_2020_MOD13A2/tifs_files/2km",ndvi))
        print("remove： {}".format(ndvi))
