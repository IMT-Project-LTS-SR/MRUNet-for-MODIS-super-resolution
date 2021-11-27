# from test_unet import *
import torch
import torch.nn as nn
from torch.nn import functional as F
from utils import *
import os
import time
###############################################
# from unet_parts import *
import cv2
import math

import sys
sys.path.insert(0,".") 
from Thunmpy import ThunmFit
import numpy as np
import matplotlib.pyplot as plt
from osgeo import gdal, gdalconst
import math
from scipy.spatial.distance import pdist, squareform
import scipy.optimize as opt
from utils import *
import os


def ATPRK(ndvi_1km, LSTd_1km, ndvi_2km, LSTd_2km):
    #FIT
    path_index = ndvi_2km
    path_temperature = LSTd_2km
    #path_index = 'TEST_Thunmpy_08-07-2019/Index/NDBI_080628_100m.dat'
    #path_temperature = 'TEST_Thunmpy_08-07-2019/Temp/LST_4_bandes_jour_satellite_100m_bruite.bsq'
    min_T = 270
    path_fit = 'Fit_NDBI_T.txt'
    plot =0
    path_plot = 'tmp_fig/NDBI_vs_T.png'

    linear_fit_test(path_index, path_temperature, min_T, path_fit, plot, path_plot)
    # index (894, 135)
    # Temp (894, 135)

    #UNMIXING
    path_index = ndvi_1km
    path_temperature = LSTd_2km
    path_fit = 'Fit_NDBI_T.txt'
    path_out = 'T_unm_1km_from2km_jourNDBI.bsq'
    iscale=2
    path_mask=0

    T_unm = linear_unmixing_test(path_index, path_temperature, path_fit, iscale,path_mask, path_out)
    # index (1787, 269)
    # Temp (358, 54)

    #CORR
    path_index = ndvi_2km
    path_fit = 'Fit_NDBI_T.txt'
    path_temperature = LSTd_2km
    path_unm = 'T_unm_1km_from2km_jourNDBI.bsq'
    iscale=2
    # scc=2
    scc=32
    block_size = 3 # 3 
    sill=7
    ran=1000
    path_out='Delta_T_1km_from2km_jour_NDBI_atprk.bsq'
    path_out1='T_unm_add_1km_from2km_jour_NDBI_atprk.bsq'
    path_out2='Delta_T_final_1km_from2km_jour_NDBI_atprk.bsq'
    path_out3='T_unmixed_final_1km_from2km_jour_NDBI_atprk.bsq'
    path_plot= 'Gamma_c_NDBI_2km.png'

    prediction = correction_ATPRK_test(path_index, path_fit, path_temperature, path_unm, iscale, scc, block_size, sill, ran, path_out, path_out1, path_out2, path_out3,path_plot,T_unm)
    return prediction, LSTd_1km



if __name__ == '__main__':
    start_time = time.time()
    root_dir = '../MODIS/'
    LST_1km_dir = os.path.join(root_dir,"MOD_2020_MOD11A1/tifs_files/1km")
    LST_2km_dir = os.path.join(root_dir,"MOD_2020_MOD11A1/tifs_files/2km")
    NDVI_1km_dir = os.path.join(root_dir,"MOD_2020_MOD13A2/tifs_files/1km")
    NDVI_2km_dir = os.path.join(root_dir,"MOD_2020_MOD13A2/tifs_files/2km")
    LST_1km_files = os.listdir(LST_1km_dir)
    LST_2km_files = os.listdir(LST_2km_dir)
    NDVI_1km_files = os.listdir(NDVI_1km_dir)
    NDVI_2km_files = os.listdir(NDVI_2km_dir)
    LST_to_NDVI_dic = np.load(os.path.join(root_dir,"LST_to_NDVI_dic.npy"), allow_pickle=1).item()


    dataset = NDVI_LST(LST_1km_dir, LST_2km_dir, NDVI_1km_dir, NDVI_2km_dir, LST_1km_files, LST_2km_files, NDVI_1km_files, NDVI_2km_files, LST_to_NDVI_dic)
    print("End, using {:.4f}s".format(time.time()-start_time))

    train_dataset, valid_dataset = torch.utils.data.random_split(dataset=dataset,lengths=[100, len(dataset)-100])
    batch_size = 64
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size)
    valid_loader = torch.utils.data.DataLoader(valid_dataset, batch_size=batch_size)

    cubic_labels = np.zeros([len(valid_dataset),64,64])
    cubic_predictions = np.zeros([len(valid_dataset),64,64])
    cubic_PSNR = np.zeros(len(valid_dataset))
    cubic_RMSE = np.zeros(len(valid_dataset))

    
    atprk_labels = np.zeros([len(valid_dataset),64,64])
    atprk_predictions = np.zeros([len(valid_dataset),64,64])
    atprk_PSNR = np.zeros(len(valid_dataset))
    atprk_RMSE = np.zeros(len(valid_dataset))
    for idx, (ndvi_1km, LSTd_1km, ndvi_2km, LSTd_2km) in enumerate(valid_dataset):
        # Test cubic
        cubic_predictions[idx,:,:] = cv2.resize(LSTd_2km, (LSTd_1km.shape), cv2.INTER_CUBIC)
        cubic_labels[idx,:,:] = LSTd_1km
        cubic_PSNR[idx], cubic_RMSE[idx] = psnr_notorch(cubic_labels[idx,:,:], cubic_predictions[idx,:,:])
        
        #%% 2. Test ATPRK
        atprk_predictions[idx,:,:], atprk_labels[idx,:,:]= ATPRK(ndvi_1km, LSTd_1km, ndvi_2km, LSTd_2km)
        atprk_PSNR[idx], atprk_RMSE[idx] = psnr_notorch(atprk_labels[idx,2:-2,2:-2], atprk_predictions[idx,2:-2,2:-2])
        print("Idx {} PSNR: {:.4f} and RMSE {:.4f}: ".format(str(idx).zfill(3),atprk_PSNR[idx], atprk_RMSE[idx]))

        fig, ax = plt.subplots(nrows=1, ncols=3, figsize=(18, 6))
        ax[0].imshow(LSTd_1km)
        ax[0].set_title("1km_gt")

        ax[1].imshow(atprk_predictions[idx,:,:])
        ax[1].set_title("1km_atprk")

        ax[2].imshow(cubic_predictions[idx,:,:])
        ax[2].set_title("1km_cubic")

        os.makedirs("tmp_fig_delete/",exist_ok=1)
        plt.savefig("tmp_fig_delete/LSTd_1km_{}.png".format(idx))
        plt.close('all')
   

    # plot and save
    fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(12, 6))
    ax[0].plot(atprk_PSNR,'r')
    ax[0].plot(cubic_PSNR,'b')
    ax[0].set_title("PSNR")
    ax[0].legend(['atprk','cubic'])

    ax[1].plot(atprk_RMSE,'r')
    ax[1].plot(cubic_RMSE,'b')
    ax[1].set_title("RMSE")
    ax[1].legend(['atprk rmse {:.4f}'.format(np.mean(atprk_RMSE)),'cubic rmse {:.4f}'.format(np.mean(cubic_RMSE))])

    os.makedirs("outputs/ATPRK/images")
    plt.savefig("outputs/ATPRK/images/Compare_ATPRK_and_Cubic.png")
    plt.show()



    # fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(6, 6))
    plt.figure()
    plt.plot(atprk_RMSE,'r')
    plt.plot(cubic_RMSE,'b')
    plt.plot(np.array(unet_RMSE),'g')
    plt.title("RMSE")
    plt.legend(['atprk rmse {:.4f}'.format(np.mean(atprk_RMSE)),\
                'cubic rmse {:.4f}'.format(np.mean(cubic_RMSE)),\
                'unet rmse {:.4f}'.format(np.mean(unet_RMSE))])
    plt.savefig("outputs/ATPRK/images/Compare_ATPRK_and_Cubic_and_unet.png")


    Compare_ATPRK_and_Cubic = {}

    Compare_ATPRK_and_Cubic["atprk_predictions"] = atprk_predictions
    Compare_ATPRK_and_Cubic["atprk_labels"] = atprk_labels
    Compare_ATPRK_and_Cubic["atprk_PSNR"] = atprk_PSNR
    Compare_ATPRK_and_Cubic["atprk_RMSE"] = atprk_RMSE

    Compare_ATPRK_and_Cubic["cubic_predictions"] = cubic_predictions
    Compare_ATPRK_and_Cubic["cubic_labels"] = cubic_labels
    Compare_ATPRK_and_Cubic["cubic_PSNR"] = cubic_PSNR
    Compare_ATPRK_and_Cubic["cubic_RMSE"] = cubic_RMSE

    np.save("outputs/ATPRK/Compare_ATPRK_and_Cubic",Compare_ATPRK_and_Cubic)