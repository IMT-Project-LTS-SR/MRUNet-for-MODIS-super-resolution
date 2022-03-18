import os
# from pymodis import downmodis
import numpy as np
import time
from utils import *
from argparse import ArgumentParser

def MODIS_Data_Preprocessing(year, product, num_threads):
    sensor        = product.split(".")[0]
    root_dir      = 'MODIS/MOD_{}_{}'.format(year,sensor)
    hdfs_path     = os.path.join(root_dir, 'hdfs_files')
    tifs_1km_path = os.path.join(root_dir, 'tifs_files/1km')
    tifs_2km_path = os.path.join(root_dir, 'tifs_files/2km')
    tifs_4km_path = os.path.join(root_dir, 'tifs_files/4km')

    os.makedirs(hdfs_path,exist_ok=1)
    os.makedirs(tifs_1km_path,exist_ok=1)
    os.makedirs(tifs_2km_path,exist_ok=1)
    os.makedirs(tifs_4km_path,exist_ok=1)

    print("start to processing {}".format(hdfs_path))
    hdfs = os.listdir(hdfs_path)
    hdfs.sort()
    start_time = time.time()
    # Core images with multi-core
    # with pymp.Parallel(num_threads) as p:
        # for index in p.range(0, len(hdfs)):

    for index in range(0, len(hdfs)):
        hdf = hdfs[index]
        if not hdf.endswith('hdf'): continue
        hdf_path = os.path.join(hdfs_path,hdf)
        if sensor=='MOD11A1':
            # try:
                crop_modis(hdf_path, hdf,tifs_1km_path, tifs_2km_path,tifs_4km_path, 64, (64,64))
            # except:
                # shutil.rmtree(hdf_path)
                # print("problem", hdf_path)
                # pass
        elif sensor=='MOD13A2':
            # try:
                crop_modis_MOD13A2(hdf_path, hdf,tifs_1km_path, tifs_2km_path,tifs_4km_path, 64, (64,64))
            # except:
                # shutil.rmtree(hdf_path)
                # print("problem", hdf_path)
                # pass

    print("Using {:.4f}s to process product = {}".format(time.time()-start_time, product))

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument('--year_begin', type=int, default=2000)
    parser.add_argument('--year_end', type=int, default=2021)
    args = parser.parse_args()

    years = list(np.arange(args.year_begin, args.year_end))
    # LST: "MOD11A1.061", NDVI: "MOD13A2.061"
    products = ["MOD11A1.061","MOD13A2.061"] 
    # tiles to download, France is in h17v04 and h18v04 , string of tiles separated by comma
    tiles = "h18v04"
    # Cores number to use     
    num_threads = 4
    for year in years:
        for product in products:
            MODIS_Data_Preprocessing(year, product, num_threads)