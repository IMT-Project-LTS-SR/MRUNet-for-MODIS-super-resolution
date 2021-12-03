import os
import glob
from pymodis import downmodis
import matplotlib.pyplot as plt
import numpy as np
from osgeo import gdal
import pymp
import time
from utils import *
from argparse import ArgumentParser


def MODIS_Parallel_Downloader(year,product, num_threads):
    sensor        = product.split(".")[0]
    hdfs_path     = 'MODIS/MOD_{}_{}/hdfs_files'.format(year,sensor)
    tifs_1km_path = 'MODIS/MOD_{}_{}/tifs_files/1km'.format(year,sensor)
    tifs_2km_path = 'MODIS/MOD_{}_{}/tifs_files/2km'.format(year,sensor)
    tifs_4km_path = 'MODIS/MOD_{}_{}/tifs_files/4km'.format(year,sensor)

    os.makedirs(hdfs_path,exist_ok=1)
    os.makedirs(tifs_1km_path,exist_ok=1)
    os.makedirs(tifs_2km_path,exist_ok=1)
    os.makedirs(tifs_4km_path,exist_ok=1)

    # Download data with multi-core
    with pymp.Parallel(num_threads) as p:
    # if 1:
        for month in p.range(1, 13):
            startdate = "{}-{}-01".format(str(year),str(month).zfill(2))
            if month != 12: 
                enddate = "{}-{}-01".format(str(year),str(month+1).zfill(2))
            else:
                enddate = "{}-{}-31".format(str(year),str(month).zfill(2))

            start_time = time.time()
            print("Start to download {} From {} to {}".format(hdfs_path, startdate,enddate))
            try:
                modisDown = downmodis.downModis(user="projet3a",password="Projet3AIMT",product=product,destinationFolder=hdfs_path, tiles=tiles, today=startdate, enddate=enddate)
                modisDown.connect()
                modisDown.downloadsAllDay()
            except:
                print("Download Error{} From {} to {}".format(hdfs_path, startdate,enddate))
            print("Finish download {} From {} to {}, time cost: {:.4f}".format(hdfs_path, startdate,enddate,time.time()-start_time))
            

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument('--year_begin', type=int, default=2017)
    parser.add_argument('--year_end', type=int, default=2017)
    args = parser.parse_args()

    years = list(np.arange(args.year_end,args.year_begin-1,-1))
    products = ["MOD13Q1.061"] # LST: "MOD11A1.061", NDVI_1km: "MOD13A2.061", NDVI_250m: "MOD13Q1.061"
    tiles = "h18v04" # tiles to download, France is in h17v04 and h18v04 , string of tiles separated by comma
    num_threads = 6 # Cores number to use 
    for year in years:
        for product in products:
            MODIS_Parallel_Downloader(year,product, num_threads)

    # python modis_downloader.py --year_begin 2017 --year_end 2020