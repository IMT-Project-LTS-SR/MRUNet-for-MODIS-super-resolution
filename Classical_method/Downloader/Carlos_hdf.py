import os
import glob
# from pymodis import downmodis
import matplotlib.pyplot as plt
import numpy as np
from osgeo import gdal
import skimage.measure
import pymp
import time
from utils import *
import shutil


year          = 2017
sensor        = 'MOD11A1'
hdfs_path     = 'tmp/MOD_{}_{}/hdfs_files'.format(year,sensor)
tifs_1km_path = 'tmp/MOD_{}_{}/tifs_files/1km'.format(year,sensor)
tifs_2km_path = 'tmp/MOD_{}_{}/tifs_files/2km'.format(year,sensor)
tifs_4km_path = 'tmp/MOD_{}_{}/tifs_files/4km'.format(year,sensor)

os.makedirs(hdfs_path,exist_ok=1)
os.makedirs(tifs_1km_path,exist_ok=1)
os.makedirs(tifs_2km_path,exist_ok=1)
os.makedirs(tifs_4km_path,exist_ok=1)

print("start to processing {}".format(hdfs_path))
hdfs = os.listdir(hdfs_path)
hdfs.sort()
start_time = time.time()
# Core images with multi-core

hdf = "MOD11A1.A2017026.h18v04.061.2021296231954.hdf"
hdf_path = os.path.join(hdfs_path,hdf)
# crop_modis(hdf_path, hdf,tifs_1km_path, 2,tifs_2km_path, 64, (64,64))
if sensor=='MOD11A1':
    crop_modis(hdf_path, hdf,tifs_1km_path, tifs_2km_path,tifs_4km_path, 64, (64,64))