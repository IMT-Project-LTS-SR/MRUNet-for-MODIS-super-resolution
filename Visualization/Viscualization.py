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
from models import ATPRK

# from Compare_ATPRK_and_Cubic import ATPRK
class UNet(nn.Module):
    def __init__(self, n_channels=1, n_classes=1, bilinear=True):
        super(UNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear

        self.inc = DoubleConv(n_channels, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        factor = 2 if bilinear else 1
        self.down4 = Down(512, 1024 // factor)
        self.up1 = Up(1024, 512 // factor, bilinear)
        self.up2 = Up(512, 256 // factor, bilinear)
        self.up3 = Up(256, 128 // factor, bilinear)
        self.up4 = Up(128, 64, bilinear)
        self.outc = OutConv(64, n_classes)
        # self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.up = nn.ConvTranspose2d(64, 64, kernel_size=2, stride=2)
        self.up_dc = DoubleConv(64, 64)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        xp = self.up1(x5, x4)
        xp = self.up2(xp, x3)
        xp = self.up3(xp, x2)
        xp = self.up4(xp, x1)
        # x = self.up(x)
        # x = self.up_dc(x)
        logits = self.outc(xp) + x
        return logits


def main(train_loader,valid_loader,batch_size):
    import matplotlib.pyplot as plt
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("Use device: {}".format(device))
    model = UNet().to(device)

    criterion = nn.MSELoss()

    os.makedirs("outputs/visualization",exist_ok=1)
    # unmixing
    model.load_state_dict(torch.load("modelUnet_batch_skip_cv2_zerocloudflip2.pth"))

    model.eval()
    unet_PSNR = []
    unet_RMSE = []

    cubic_labels = np.zeros([len(valid_dataset),64,64])
    cubic_predictions = np.zeros([len(valid_dataset),64,64])
    cubic_PSNR = np.zeros(len(valid_dataset))
    cubic_RMSE = np.zeros(len(valid_dataset))
    
    atprk_labels = np.zeros([len(valid_dataset),64,64])
    atprk_predictions = np.zeros([len(valid_dataset),64,64])
    atprk_PSNR = np.zeros(len(valid_dataset))
    atprk_RMSE = np.zeros(len(valid_dataset))

    with torch.no_grad():
        for ndvi_1km, LST_K_day_1km, ndvi_2km, LST_K_day_2km in valid_loader:
            bicubic = np.zeros_like(LST_K_day_1km)
            for idx in range(bicubic.shape[0]):
                bicubic[idx,:,:] = cv2.resize(LST_K_day_2km[idx,:,:].numpy(), (LST_K_day_1km[idx,:,:].numpy().shape), cv2.INTER_CUBIC)

            LST_K_day_1km = LST_K_day_1km.unsqueeze(1).float().to(device)
            LST_K_day_2km = LST_K_day_2km.unsqueeze(1).float().to(device)
            bicubic = torch.from_numpy(bicubic/324.260009765625)
            bicubic = bicubic.unsqueeze(1).float().to(device)

            # Forward pass
            outputs = model(bicubic) * 324.260009765625
            loss = criterion(outputs, LST_K_day_1km)

            for batch_idx in range(bicubic.shape[0]):
                # test unet
                unet_PSNR_, unet_RMSE_ = psnr_notorch(LST_K_day_1km[batch_idx].squeeze(0).cpu().numpy(), outputs[batch_idx].squeeze(0).cpu().numpy())

                unet_PSNR.append(unet_PSNR_)
                unet_RMSE.append(unet_RMSE_)
                print("Idx {} PSNR: {:.4f} and RMSE {:.4f}: ".format(str(batch_idx).zfill(3),unet_PSNR_, unet_RMSE_))

                # 1. Test cubic
                cubic_predictions[batch_idx,:,:] = cv2.resize(LST_K_day_2km[batch_idx,:,:].squeeze(0).cpu().numpy(), (LST_K_day_1km[batch_idx,:,:].squeeze(0).cpu().numpy().shape), cv2.INTER_CUBIC)
                cubic_labels[batch_idx,:,:] = LST_K_day_1km[batch_idx,:,:].squeeze(0).cpu().numpy()
                cubic_PSNR[batch_idx], cubic_RMSE[batch_idx] = psnr_notorch(cubic_labels[batch_idx,:,:], cubic_predictions[batch_idx,:,:])
                
                #%% 2. Test ATPRK
                atprk_predictions[batch_idx,:,:], atprk_labels[batch_idx,:,:], Delta_T_final, TT_unm= ATPRK(ndvi_1km[batch_idx,:,:].squeeze(0).cpu().numpy(), \
                                                                                    LST_K_day_1km[batch_idx,:,:].squeeze(0).cpu().numpy(),\
                                                                                    ndvi_2km[batch_idx,:,:].squeeze(0).cpu().numpy(), \
                                                                                    LST_K_day_2km[batch_idx,:,:].squeeze(0).cpu().numpy())
                atprk_PSNR[batch_idx], atprk_RMSE[batch_idx] = psnr_notorch(atprk_labels[batch_idx,2:-2,2:-2], atprk_predictions[batch_idx,2:-2,2:-2])

                
                fig, ax = plt.subplots(nrows=2, ncols=2, figsize=(12, 12))
                ax[0,0].imshow(LST_K_day_1km[batch_idx,:,:].squeeze(0).cpu().numpy())
                ax[0,0].set_title("1km_LST")
                ax[0,1].imshow(Delta_T_final)
                ax[0,1].set_title("Delta_T_final")
                ax[1,0].imshow(TT_unm)
                ax[1,0].set_title("TT_unm")
                ax[1,1].imshow(atprk_predictions[batch_idx,:,:])
                ax[1,1].set_title("1km_atprk rmse {:.4f}".format(atprk_RMSE[batch_idx]))

                os.makedirs("outputs/visualization/ATPRK",exist_ok=1)
                plt.savefig("outputs/visualization/ATPRK/LST_K_day_1km_{}.png".format(batch_idx))

                fig, ax = plt.subplots(nrows=2, ncols=3, figsize=(18, 12))
                ax[0,0].imshow(LST_K_day_1km[batch_idx,:,:].squeeze(0).cpu().numpy())
                ax[0,0].set_title("1km_LST")
                ax[0,1].imshow(LST_K_day_2km[batch_idx,:,:].squeeze(0).cpu().numpy())
                ax[0,1].set_title("2km_LST")
                ax[0,2].imshow(ndvi_1km[batch_idx,:,:].squeeze(0).cpu().numpy())
                ax[0,2].set_title("1km_NDVI")
                ax[1,0].imshow(atprk_predictions[batch_idx,:,:])
                ax[1,0].set_title("1km_atprk rmse {:.4f}".format(atprk_RMSE[batch_idx]))
                ax[1,1].imshow(cubic_predictions[batch_idx,:,:])
                ax[1,1].set_title("1km_cubic rmse {:.4f}".format(cubic_RMSE[batch_idx]))

                ax[1,2].imshow(outputs[batch_idx].squeeze(0).cpu().numpy())
                ax[1,2].set_title("1km_unet rmse {:.4f}".format(unet_RMSE_))


                os.makedirs("outputs/visualization/",exist_ok=1)
                plt.savefig("outputs/visualization/LST_K_day_1km_{}.png".format(batch_idx))

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
    print("Start")
    setup_seed(20)
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
    num_workers = 0
    batch_size = 512

    dataset = NDVI_LST(LST_1km_dir, LST_2km_dir, NDVI_1km_dir, NDVI_2km_dir, LST_1km_files, LST_2km_files, NDVI_1km_files, NDVI_2km_files, LST_to_NDVI_dic)

    print("End, using {:.4f}s".format(time.time()-start_time))
    
    train_dataset, valid_dataset = torch.utils.data.random_split(dataset=dataset,lengths=[100, len(dataset)-100])
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size,shuffle=True, num_workers= num_workers)
    valid_loader = torch.utils.data.DataLoader(valid_dataset, batch_size=batch_size,shuffle=False, num_workers= num_workers)
    
    unet_PSNR, unet_RMSE, cubic_RMSE, atprk_RMSE = main(train_loader,valid_loader,batch_size)
