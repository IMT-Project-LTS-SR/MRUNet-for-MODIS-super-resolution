import torch
import torch.nn as nn
from torch.nn import functional as F
from utils import *
import os
import time
from torchvision import transforms
###############################################
from unet_parts import *
from methods.ATPRK import ATPRK
from methods.models import UNet, Encoder_Decoder, UNet_Minh, UNet_Channel_Fusion, UNet_Channel_Fusion_Cubic_Inputs, UNet_Channel_Fusion_Cubic_Inputs_Add
import cv2
import sys


def main(model_name,train_loader,valid_loader,batch_size, max_all, min_all,valid_lengt):
    # https://bbs.cvmart.net/articles/5353
    import matplotlib.pyplot as plt
    from torch.utils.tensorboard import SummaryWriter
    
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

    # model_name = model._get_name()

    chkpot_path = os.path.join("outputs/MSE_Grad_loss/{}/chkpts/".format(model_name))
    images_path = os.path.join("outputs/MSE_Grad_loss/{}/images/".format(model_name))
    logevs_path = os.path.join("/homes/g19tian/Documents/outputs/1_projet3A/MSE_Grad_loss/{}/logevs/".format(model_name))

    os.makedirs(chkpot_path,exist_ok=1)
    os.makedirs(images_path,exist_ok=1)
    os.makedirs(logevs_path,exist_ok=1)

    writer = SummaryWriter(logevs_path)

    # set training parameters
    num_epochs = 4000
    learning_rate = 0.1
    
    # Loss and optimizer
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=50, threshold=0.0001, threshold_mode='rel', cooldown=0, min_lr=1e-10, eps=1e-10, verbose=True)
    # scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=300, gamma=0.5)
    # Train the model
    total_step = len(train_loader)
    train_epochs_loss = []
    valid_epochs_loss = []

    model.load_state_dict(torch.load("outputs/MSE_Grad_loss/UNet_Minh/model_epoch_0400_loss_5.4516.pt"))
    for epoch in range(num_epochs): # refer to https://zhuanlan.zhihu.com/p/104019160
        
        start_time = time.time()
        loss_epoch = 10000
        train_epoch_loss = []
        valid_epoch_loss = []
        model.train()
        for i, (ndvi_1km, LSTd_1km, ndvi_2km, LSTd_2km) in enumerate(train_loader):
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
            # inputs = torch.cat([LSTd_2km,ndvi_2km],dim=1)
            
            if model_name == "UNet_Minh":    
                inputs = bicubic_LSTd # 64*64
            elif model_name == "UNet_Channel_Fusion_Cubic_Inputs":
                inputs = torch.cat([bicubic_LSTd,bicubic_NDVI],dim=1) # 64*64
            elif model_name == "UNet_Channel_Fusion_Cubic_Inputs_Add":
                inputs = torch.cat([bicubic_LSTd,torch.add(bicubic_LSTd,bicubic_NDVI)],dim=1) # 64*64
            elif model_name == "Encoder_Decoder":
                # inputs = LSTd_2km
                inputs = torch.cat([bicubic_LSTd,bicubic_NDVI],dim=1) # 64*64
                

            # outputs = model(inputs)*(max_all["LSTd_2km"]-min_all["LSTd_2km"])+min_all["LSTd_2km"]
            outputs = model(inputs)*max_all["LSTd_2km"]

            # loss = criterion(outputs, LSTd_1km)
            loss = get_grad_loss(outputs, LSTd_1km)

            # Backward and optimizer
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_epoch_loss.append(loss.item())
            
            if i+1 == len(train_loader):
                print('Train Epoch: [{}/{}], Step: [{}/{}], Loss: {}, Time: {:.4f}s/{:.4f}h'.format(str(epoch+1).zfill(4), num_epochs, i+1, total_step, loss.item(),time.time() - start_time, (time.time() - start_time)*num_epochs/60/60))
                start_time = time.time()

        # Validation
        model.eval()
        i = 0
        start_time = time.time()
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

        # 不记录模型梯度信息
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

                # loss = criterion(outputs, LSTd_1km)
                loss = get_grad_loss(outputs, LSTd_1km)
                # valid_epoch_loss.append(loss.item())
                # if np.mean(valid_epoch_loss)<10:
                #     for batch_idx in range(batch_size):
                #         if batch_idx>=outputs.shape[0]:break
                #         unet_PSNR_, unet_RMSE_ = psnr_notorch(LSTd_1km[batch_idx].squeeze(0).cpu().numpy(), outputs[batch_idx].squeeze(0).cpu().numpy())
                #         unet_PSNR.append(unet_PSNR_)
                #         unet_RMSE.append(unet_RMSE_)
                #     # print("Idx {} PSNR: {:.4f} and RMSE {:.4f}: ".format(str(batch_idx).zfill(3),unet_PSNR_, unet_RMSE_))

                if i+1 == len(valid_loader):
                    print('Valid Epoch: [{}/{}], Step: [{}/{}], Loss: {}, Time: {:.4f}s/{:.4f}h'.format(str(epoch+1).zfill(4), num_epochs, i+1, total_step, loss.item(),time.time() - start_time, (time.time() - start_time)*num_epochs/60/60))
                    start_time = time.time()

            # Visual
            if epoch % 500 == 0:
            # if epoch % 500 == 0 and epoch>100:
                for batch_idx in range(int(bicubic_LSTd.shape[0]/10)):
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
                    
                    fig, ax = plt.subplots(nrows=2, ncols=3, figsize=(18, 12))
                    ax[0,0].imshow(LSTd_1km[batch_idx,:,:].squeeze(0).cpu().numpy())
                    ax[0,0].set_title("1km_LST")
                    ax[0,1].imshow(LSTd_2km[batch_idx,:,:].squeeze(0).cpu().numpy())
                    ax[0,1].set_title("2km_LST")
                    ax[0,2].imshow(ndvi_1km[batch_idx,:,:].squeeze(0).cpu().numpy())
                    ax[0,2].set_title("1km_NDVI")
                    ax[1,0].imshow(atprk_predictions[batch_idx,:,:])
                    ax[1,0].set_title("1km_atprk rmse {:.4f}".format(atprk_RMSE[batch_idx]))
                    # ax[1,0].imshow(cubic_predictions[batch_idx,:,:])
                    # ax[1,0].set_title("1km_cubic rmse {:.4f}".format(cubic_RMSE[batch_idx]))
                    ax[1,1].imshow(cubic_predictions[batch_idx,:,:])
                    ax[1,1].set_title("1km_cubic rmse {:.4f}".format(cubic_RMSE[batch_idx]))

                    ax[1,2].imshow(outputs[batch_idx].squeeze(0).cpu().numpy())
                    ax[1,2].set_title("1km_unet rmse {:.4f}".format(unet_RMSE_))

                    plt.savefig(os.path.join(images_path,"Epoch_{}_Comparaison_{}.png".format(str(epoch).zfill(5),str(batch_idx).zfill(3))))

        

        train_epochs_loss.append(np.mean(train_epoch_loss))
        valid_epochs_loss.append(np.mean(valid_epoch_loss))
        writer.add_scalar('Train/Loss', np.mean(train_epoch_loss), epoch)
        writer.add_scalar('Valid/Loss', np.mean(valid_epoch_loss), epoch)
        writer.add_scalar('RMSE/Loss', np.mean(unet_RMSE), epoch)
        # writer.add_scalar('lr/epoch', scheduler._last_lr[-1], epoch)
        writer.flush()
        scheduler.step(np.mean(valid_epoch_loss))


        if epoch%200 ==0 and epoch!=0:
            print('===Epoch: [{}/{}], Loss: {}, Time: {:.4f}'.format(str(epoch+1).zfill(4), num_epochs, np.mean(train_epoch_loss),time.time() - start_time))
            # torch.save(model.state_dict(), "model_lr_0_1_f/model_best.pt")
            torch.save(model.state_dict(), os.path.join(chkpot_path,"model_epoch_{}_loss_{:.4f}.pt".format(str(epoch).zfill(4),np.mean(train_epoch_loss))))





if __name__ == '__main__':
    setup_seed(20)
    if len(sys.argv)>1:
        model_name = sys.argv[1]
    else:
        model_name = "UNet_Minh"

    train_loader,valid_loader,batch_size, max_all, min_all, valid_lengt = init()
    unet_PSNR, unet_RMSE, cubic_RMSE, atprk_RMSE = main(model_name,train_loader,valid_loader,batch_size, max_all, min_all, valid_lengt)

    # nohup python -u train.py UNet_Minh > outputs/MSE_Grad_loss/UNet_Minh/UNet_Minh 2>&1 &
    # nohup python -u train.py UNet_Channel_Fusion_Cubic_Inputs > outputs/MSE_Grad_loss/UNet_Channel_Fusion_Cubic_Inputs/UNet_Channel_Fusion_Cubic_Inputs 2>&1 &
    # nohup python -u train.py UNet_Channel_Fusion_Cubic_Inputs_Add > outputs/MSE_Grad_loss/UNet_Channel_Fusion_Cubic_Inputs_Add/UNet_Channel_Fusion_Cubic_Inputs_Add 2>&1 &
    # nohup python -u train.py Encoder_Decoder > outputs/MSE_Grad_loss/Encoder_Decoder/Encoder_Decoder 2>&1 &