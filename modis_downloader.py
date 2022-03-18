import os
from pymodis import downmodis
import numpy as np
import pymp
import time
from utils import *
from argparse import ArgumentParser


def MODIS_Parallel_Downloader(year,product, num_threads, user="projet3a", password="Projet3AIMT"):
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
                modisDown = downmodis.downModis(user=user,password=password,product=product,destinationFolder=hdfs_path, tiles=tiles, today=startdate, enddate=enddate)
                modisDown.connect()
                modisDown.downloadsAllDay()
                # modisDown.downloadsAllDay(clean=True, allDays=True)
            except:
                print("Download Error {} From {} to {}".format(hdfs_path, startdate,enddate))
            print("Finish download {} From {} to {}, time cost: {:.4f}".format(hdfs_path, startdate,enddate,time.time()-start_time))
            

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument('--year_begin', type=int, default=2000)
    parser.add_argument('--year_end', type=int, default=2021)
    parser.add_argument('--username', type=str, default="ganglina")
    parser.add_argument('--password', type=str, default="505166Tgl")
    args = parser.parse_args()

    years = list(np.arange(args.year_begin, args.year_end))
    products = ["MOD11A1.061", "MOD13A2.061"] # LST: "MOD11A1.061", NDVI_1km: "MOD13A2.061", NDVI_250m: "MOD13Q1.061"
    tiles = "h18v04" # tiles to download, France is in h17v04 and h18v04 , string of tiles separated by comma
    num_threads = 6 # Cores number to use 
    for year in years:
        for product in products:
            MODIS_Parallel_Downloader(year,product, num_threads, args.username, args.password)


    # python modis_downloader.py --username <username> --password <password> 