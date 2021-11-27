import os
import glob
from pymodis import downmodis
import matplotlib.pyplot as plt
import numpy as np
from osgeo import gdal
import skimage.measure
import pymp
import time
from utils import *
import shutil

def MODIS_Data_Preprocessing(year,product, num_threads):
    sensor        = product.split(".")[0]
    hdfs_path     = 'MODIS/MOD_{}_{}/hdfs_files'.format(year,sensor)
    tifs_1km_path = 'MODIS/MOD_{}_{}/tifs_files/1km'.format(year,sensor)
    tifs_2km_path = 'MODIS/MOD_{}_{}/tifs_files/2km'.format(year,sensor)
    tifs_4km_path = 'MODIS/MOD_{}_{}/tifs_files/4km'.format(year,sensor)

    os.makedirs(hdfs_path,exist_ok=1)
    os.makedirs(tifs_1km_path,exist_ok=1)
    os.makedirs(tifs_2km_path,exist_ok=1)
    os.makedirs(tifs_4km_path,exist_ok=1)

    print("start to processing {}".format(hdfs_path))
    hdfs = os.listdir(hdfs_path)
    hdfs.sort()
    start_time = time.time()
    # Core images with multi-core
    with pymp.Parallel(num_threads) as p:
        for index in p.range(0, len(hdfs)):
    # if 1:
        # for index in range(0, len(hdfs)):

            hdf = hdfs[index]
            if not hdf.endswith('hdf'): continue
            hdf_path = os.path.join(hdfs_path,hdf)
            crop_modis(hdf_path, hdf,tifs_1km_path, tifs_2km_path, 64, (64,64))
        print("Using {:.4f}s to process data".format(time.time()-start_time))
        # print("There are {} tif files left".format(len(os.listdir(tifs_1km_path))))


    # shutil.rmtree(hdfs_path)

if __name__ == "__main__":
    years = list(np.arange(2000,2021))
    products = ["MOD11A1.061", "MOD13A2.061"] # LST: "MOD11A1.061", NDVI: "MOD13A2.061"
    tiles = "h18v04" # tiles to download, France is in h17v04 and h18v04 , string of tiles separated by comma
    num_threads = 6 # Cores number to use 
    for year in years:
        for product in products:
            MODIS_Data_Preprocessing(year,product, num_threads)