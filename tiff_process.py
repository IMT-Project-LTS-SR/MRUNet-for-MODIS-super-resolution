import pymp
from multiprocessing import Pool, cpu_count, Process
import time
import os
from osgeo import gdal

def tiff_process(data_path):
    count = 0
    cores = 4

    start = time.time()
    tifs = os.listdir(data_path)
    Y_day = pymp.shared.array((len(tifs),64,64), dtype='float64')
    Y_night = pymp.shared.array((len(tifs),64,64), dtype='float64')

    with pymp.Parallel(cores) as p:
      for idx in p.range(len(tifs)):
        tif = tifs[idx]
        if tif.endswith('tif'):
          tif_path = os.path.join(data_path,tif)
          LST_K_day,LST_K_night,cols,rows,projection,geotransform = read_tif(tif_path)
          Y_day[idx,:,:] = LST_K_day
          Y_night[idx,:,:] = LST_K_night       


    end = time.time()
    print(f"Process ended in {((end-start)/60):.3f} minutes")
    return Y_day, Y_night
