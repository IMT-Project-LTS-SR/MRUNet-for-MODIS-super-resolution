import torch
import torch.nn as nn
from torch.nn import functional as F
from utils import *
import os
import time
from torchvision import transforms
import cv2
###############################################
from unet_parts import *
from dataloader import NDVI_LST
from methods.ATPRK import ATPRK
from methods.models import UNet, Encoder_Decoder, UNet_Minh, UNet_Channel_Fusion, UNet_Channel_Fusion_Cubic_Inputs, UNet_Channel_Fusion_Cubic_Inputs_Add
import sys


def main(model_name,train_loader,valid_loader,batch_size, max_all, min_all,valid_lengt):
    import matplotlib.pyplot as plt
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("Use device: {}".format(device))

    use_sigmoid = 0
    if model_name == "UNet_Minh":    
        model = UNet_Minh(bilinear=False).to(device)
    elif model_name == "UNet_Channel_Fusion":
        model = UNet_Channel_Fusion(n_channels=2,bilinear=False).to(device)
    elif model_name == "UNet_Channel_Fusion_Cubic_Inputs":
        model = UNet_Channel_Fusion_Cubic_Inputs(n_channels=2,bilinear=False,use_sigmoid=use_sigmoid).to(device)
    elif model_name == "UNet_Channel_Fusion_Cubic_Inputs_Add":
        model = UNet_Channel_Fusion_Cubic_Inputs_Add(n_channels=1,bilinear=False,use_sigmoid=use_sigmoid).to(device)
    elif model_name == "Encoder_Decoder":
        model = Encoder_Decoder(18, False).to(device)

    images_path = os.path.join("outputs/MSE_Grad_loss/{}/images/".format(model_name))
    os.makedirs(images_path,exist_ok=1)

    # unmixing
    model.load_state_dict(torch.load(os.path.join(images_path,"../model_best.pt")))

    model.eval()
    unet_PSNR = []
    unet_RMSE = []

    cubic_labels = np.zeros([valid_lengt,64,64])
    cubic_predictions = np.zeros([valid_lengt,64,64])
    cubic_PSNR = np.zeros(valid_lengt)
    cubic_RMSE = np.zeros(valid_lengt)
    
    atprk_labels = np.zeros([valid_lengt,64,64])
    atprk_predictions = np.zeros([valid_lengt,64,64])
    atprk_PSNR = np.zeros(valid_lengt)
    atprk_RMSE = np.zeros(valid_lengt)

    with torch.no_grad():
        for i, (ndvi_1km, LSTd_1km, ndvi_2km, LSTd_2km) in enumerate(valid_loader):
            bicubic_LSTd = np.zeros_like(LSTd_1km)
            bicubic_NDVI = np.zeros_like(ndvi_1km)
            for idx in range(bicubic_LSTd.shape[0]):
                bicubic_LSTd[idx,:,:] = cv2.resize(LSTd_2km[idx,:,:].numpy(), (LSTd_1km[idx,:,:].numpy().shape), cv2.INTER_CUBIC)
                bicubic_NDVI[idx,:,:] = cv2.resize(ndvi_2km[idx,:,:].numpy(), (ndvi_1km[idx,:,:].numpy().shape), cv2.INTER_CUBIC)
            
            ndvi_1km = ndvi_1km.unsqueeze(1).float().to(device)
            ndvi_2km = ndvi_2km.unsqueeze(1).float().to(device)
            LSTd_1km = LSTd_1km.unsqueeze(1).float().to(device)
            LSTd_2km = LSTd_2km.unsqueeze(1).float().to(device)

            bicubic_LSTd = torch.from_numpy(bicubic_LSTd).unsqueeze(1).float().to(device)
            bicubic_NDVI = torch.from_numpy(bicubic_NDVI).unsqueeze(1).float().to(device)

            # LSTd_2km = (LSTd_2km-min_all["LSTd_2km"])/(max_all["LSTd_2km"]-min_all["LSTd_2km"])
            # ndvi_2km = (ndvi_2km-min_all["ndvi_2km"])/(max_all["ndvi_2km"]-min_all["ndvi_2km"])
            # bicubic_LSTd = (bicubic_LSTd-min_all["LSTd_1km"])/(max_all["LSTd_1km"]-min_all["LSTd_1km"])
            # bicubic_NDVI = (bicubic_NDVI-min_all["ndvi_1km"])/(max_all["ndvi_1km"]-min_all["ndvi_1km"])

            LSTd_2km = LSTd_2km/max_all["LSTd_2km"]
            ndvi_2km = ndvi_2km/max_all["ndvi_2km"]
            bicubic_LSTd = bicubic_LSTd/max_all["LSTd_1km"]
            bicubic_NDVI = bicubic_NDVI/max_all["ndvi_1km"]


            # Forward pass
            if model_name == "UNet_Minh":    
                inputs = bicubic_LSTd # 64*64
            elif model_name == "UNet_Channel_Fusion_Cubic_Inputs":
                inputs = torch.cat([bicubic_LSTd,bicubic_NDVI],dim=1) # 64*64
            elif model_name == "UNet_Channel_Fusion_Cubic_Inputs_Add":
                inputs = torch.cat([bicubic_LSTd,torch.add(bicubic_LSTd,bicubic_NDVI)],dim=1) # 64*64
            elif model_name == "Encoder_Decoder":
                # inputs = LSTd_2km
                inputs = torch.cat([bicubic_LSTd,bicubic_NDVI],dim=1) # 64*64

            # Forward pass
            # outputs = model(inputs)*(max_all["LSTd_2km"]-min_all["LSTd_2km"])+min_all["LSTd_2km"]
            # LSTd_2km = LSTd_2km*(max_all["LSTd_2km"]-min_all["LSTd_2km"])+min_all["LSTd_2km"]
            # ndvi_2km = ndvi_2km*(max_all["ndvi_2km"]-min_all["ndvi_2km"])+min_all["ndvi_2km"]
            
            outputs = model(inputs)*max_all["LSTd_2km"]
            LSTd_2km = LSTd_2km*max_all["LSTd_2km"]
            ndvi_2km = ndvi_2km*max_all["ndvi_2km"]


            for batch_idx in range(bicubic_LSTd.shape[0]):
                # test unet
                unet_PSNR_, unet_RMSE_ = psnr_notorch(LSTd_1km[batch_idx].squeeze(0).cpu().numpy(), outputs[batch_idx].squeeze(0).cpu().numpy())

                unet_PSNR.append(unet_PSNR_)
                unet_RMSE.append(unet_RMSE_)
                print("Idx {} PSNR: {:.4f} and RMSE {:.4f}: ".format(str(batch_idx).zfill(3),unet_PSNR_, unet_RMSE_))

                # 1. Test cubic
                cubic_predictions[batch_idx,:,:] = cv2.resize(LSTd_2km[batch_idx,:,:].squeeze(0).cpu().numpy(), (LSTd_1km[batch_idx,:,:].squeeze(0).cpu().numpy().shape), cv2.INTER_CUBIC)
                cubic_labels[batch_idx,:,:] = LSTd_1km[batch_idx,:,:].squeeze(0).cpu().numpy()
                cubic_PSNR[batch_idx], cubic_RMSE[batch_idx] = psnr_notorch(cubic_labels[batch_idx,:,:], cubic_predictions[batch_idx,:,:])
                
                # 2. Test ATPRK
                atprk_predictions[batch_idx,:,:], atprk_labels[batch_idx,:,:] = ATPRK(ndvi_1km[batch_idx,:,:].squeeze(0).cpu().numpy(), \
                                                                                    LSTd_1km[batch_idx,:,:].squeeze(0).cpu().numpy(),\
                                                                                    ndvi_2km[batch_idx,:,:].squeeze(0).cpu().numpy(), \
                                                                                    LSTd_2km[batch_idx,:,:].squeeze(0).cpu().numpy())
                atprk_PSNR[batch_idx], atprk_RMSE[batch_idx] = psnr_notorch(atprk_labels[batch_idx,2:-2,2:-2], atprk_predictions[batch_idx,2:-2,2:-2])
                
                # fig, ax = plt.subplots(nrows=2, ncols=3, figsize=(18, 12))
                # ax[0,0].imshow(LSTd_1km[batch_idx,:,:].squeeze(0).cpu().numpy())
                # ax[0,0].set_title("1km_LST")
                # ax[0,1].imshow(LSTd_2km[batch_idx,:,:].squeeze(0).cpu().numpy())
                # ax[0,1].set_title("2km_LST")
                # ax[0,2].imshow(ndvi_1km[batch_idx,:,:].squeeze(0).cpu().numpy())
                # ax[0,2].set_title("1km_NDVI")
                # ax[1,0].imshow(atprk_predictions[batch_idx,:,:])
                # ax[1,0].set_title("1km_atprk rmse {:.4f}".format(atprk_RMSE[batch_idx]))
                # # ax[1,0].imshow(cubic_predictions[batch_idx,:,:])
                # # ax[1,0].set_title("1km_cubic rmse {:.4f}".format(cubic_RMSE[batch_idx]))
                # ax[1,1].imshow(cubic_predictions[batch_idx,:,:])
                # ax[1,1].set_title("1km_cubic rmse {:.4f}".format(cubic_RMSE[batch_idx]))

                # ax[1,2].imshow(outputs[batch_idx].squeeze(0).cpu().numpy())
                # ax[1,2].set_title("1km_unet rmse {:.4f}".format(unet_RMSE_))

                # plt.savefig(os.path.join(images_path,"Epoch_{}_Comparaison_{}.png".format(str(epoch).zfill(5),str(batch_idx).zfill(3))))

                plt.figure()
                plt.plot(atprk_RMSE,'r')
                plt.plot(cubic_RMSE,'b')
                plt.plot(np.array(unet_RMSE),'g')
                plt.title("RMSE")
                plt.legend(['atprk rmse {:.4f}'.format(np.mean(atprk_RMSE)),\
                    'cubic rmse {:.4f}'.format(np.mean(cubic_RMSE)),\
                    'unet rmse {:.4f}'.format(np.mean(unet_RMSE))])
                plt.savefig("Compare_ATPRK_and_Cubic_and_unet.png")
                plt.close('all')

    return unet_PSNR, unet_RMSE, cubic_RMSE, atprk_RMSE


if __name__ == '__main__':
    setup_seed(20)
    if len(sys.argv)>1:
        model_name = sys.argv[1]
    else:
        model_name = "UNet_Minh"

    train_loader,valid_loader,batch_size, max_all, min_all, valid_lengt = init()
    unet_PSNR, unet_RMSE, cubic_RMSE, atprk_RMSE = main(model_name,train_loader,valid_loader,batch_size, max_all, min_all, valid_lengt)


    # python Viscualization_fusion.py UNet_Minh
    # python Viscualization_fusion.py UNet_Channel_Fusion_Cubic_Inputs
    # python Viscualization_fusion.py UNet_Channel_Fusion_Cubic_Inputs_Add