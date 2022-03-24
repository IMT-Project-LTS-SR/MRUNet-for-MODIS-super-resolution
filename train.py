from model import MRUNet
from utils import *
from tiff_process import tiff_process
from dataset import LOADDataset
import pymp
import cv2
import numpy as np
from torch.utils.data import DataLoader
from tqdm import tqdm
import torch
import torch.nn.functional as F
import torch.optim as optim
import time
import argparse
import os

parser = argparse.ArgumentParser(description="PyTorch MR UNet training from tif files contained in a data folder",
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--datapath', help='path to directory containing training tif data')
parser.add_argument('--lr', default=0.001, type=float, help='learning rate')
parser.add_argument('--epochs', default=300, type=int, help='number of epochs')
parser.add_argument('--batch_size', default=24, type=int, help='size of batch')
parser.add_argument('--model_name', type=str, help='name of the model')
parser.add_argument('--continue_train', choices=['True', 'False'], default='False', type=str, 
                    help="flag for continue training, if True - continue training the 'model_name' model, else - training from scratch")
args = parser.parse_args()


device = 'cuda' if torch.cuda.is_available() else 'cpu'
model = MRUNet(res_down=True, n_resblocks=1, bilinear=0).to(device)

scale = 4
core = 4
# Tiff process
Y_day, Y_night = tiff_process(args.datapath)
Y = np.concatenate((Y_day, Y_night),axis=0)
Y_new = pymp.shared.list()

# Eliminate images with cloud/sea pixels
with pymp.Parallel(10) as p:
  for i in p.range(Y.shape[0]):
    if len(Y[i][Y[i] == 0]) <= 0:
      Y_new.append(Y[i])
Y_new = np.array(Y_new)
start = time.time()

# Create training and validating sets
np.random.seed(1)
np.random.shuffle(Y_new) # Shuffle dataset
ratio = 0.75 # Proportion of training data
y_train = Y_new[:int(Y_new.shape[0]*ratio)]
y_val = Y_new[int(Y_new.shape[0]*ratio):]

# DATA AUGMENTATION FOR TRAINING LABEL
y_train_new = pymp.shared.list()
with pymp.Parallel(core) as p:
  for i in p.range(y_train.shape[0]):
    y_train_new.append(y_train[i])
    y_train_new.append(np.flip(y_train[i], 1))
y_train = np.array(y_train_new)
max_val = np.max(y_train) # MAX VALUE IN TRAINING SET, WHICH IS USED FOR NORMALIZATION
print('Max pixel value of training set is {},\nIMPORTANT: Please save it for later used as the normalization factor\n'.format(max_val))

# INIT TRAINING DATA AND TESTING DATA SHAPE
x_train = pymp.shared.array((y_train.shape))
x_val = pymp.shared.array((y_val.shape))

# PREPROCESS TO CREATE BICUBIC VERSION FOR MODEL INPUT
with pymp.Parallel(core) as p:
  for i in p.range(y_train.shape[0]):
    y_tr = y_train[i]
    a = downsampling(y_tr, scale)
    x_train[i,:,:] = upsampling(a , scale)
    x_train[i,:,:] = normalization(x_train[i,:,:], max_val)

with pymp.Parallel(core) as p:
  for i in p.range(y_val.shape[0]):
    y_te = y_val[i]
    a = downsampling(y_te, scale)
    x_val[i,:,:] = upsampling(a , scale)
    x_val[i,:,:] = normalization(x_val[i,:,:], max_val)

x_train = x_train.reshape((x_train.shape[0], 1, x_train.shape[1], x_train.shape[2]))
x_val = x_val.reshape((x_val.shape[0], 1, x_val.shape[1], x_val.shape[2]))
y_train = y_train.reshape((y_train.shape[0], 1, y_train.shape[1], y_train.shape[2]))
y_val = y_val.reshape((y_val.shape[0], 1, y_val.shape[1], y_val.shape[2]))
end = time.time()
print(f"Finished processing data in additional {((end-start)/60):.3f} minutes \n")

def train(model, dataloader, optimizer, train_data, max_val):
    # Train model
    model.train()
    running_loss = 0.0
    running_psnr = 0.0
    running_ssim = 0.0
    for bi, data in tqdm(enumerate(dataloader), total=int(len(train_data)/dataloader.batch_size)):
        image_data = data[0].to(device)
        label = data[1].to(device)
        
        # zero grad the optimizer
        optimizer.zero_grad()
        outputs = model(image_data)
        loss = get_loss(outputs*max_val, label)
        # backpropagation
        loss.backward()
        # update the parameters
        optimizer.step()
        # add loss of each item (total items in a batch = batch size)
        running_loss += loss.item()
        # calculate batch psnr (once every `batch_size` iterations)
        batch_psnr =  psnr(label, outputs, max_val)
        running_psnr += batch_psnr
        batch_ssim =  ssim(label, outputs, max_val)
        running_ssim += batch_ssim
    final_loss = running_loss/len(dataloader.dataset)
    final_psnr = running_psnr/int(len(train_data)/dataloader.batch_size)
    final_ssim = running_ssim/int(len(train_data)/dataloader.batch_size)
    return final_loss, final_psnr, final_ssim

def validate(model, dataloader, epoch, val_data, max_val):
    model.eval()
    running_loss = 0.0
    running_psnr = 0.0
    running_ssim = 0.0
    with torch.no_grad():
        for bi, data in tqdm(enumerate(dataloader), total=int(len(val_data)/dataloader.batch_size)):
            image_data = data[0].to(device)
            label = data[1].to(device)
            outputs = model(image_data)
            
            loss = get_loss(outputs*max_val, label)
            # add loss of each item (total items in a batch = batch size) 
            running_loss += loss.item()
            # calculate batch psnr (once every `batch_size` iterations)
            batch_psnr = psnr(label, outputs, max_val)
            running_psnr += batch_psnr
            batch_ssim =  ssim(label, outputs, max_val)
            running_ssim += batch_ssim
        outputs = outputs.cpu()
        # save_image(outputs, f"../outputs/val_sr{epoch}.png")
    final_loss = running_loss/len(dataloader.dataset)
    final_psnr = running_psnr/int(len(val_data)/dataloader.batch_size)
    final_ssim = running_ssim/int(len(val_data)/dataloader.batch_size)
    return final_loss, final_psnr, final_ssim

def main():
    # Load dataset and create data loader
    transform = None
    train_data = LOADDataset(x_train, y_train, transform=transform)
    val_data = LOADDataset(x_val, y_val, transform=transform)
    batch_size = args.batch_size
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle = True)
    val_loader = DataLoader(val_data, batch_size=batch_size)
    print('Length of training set: {} \n'.format(len(train_data)))
    print('Length of validating set: {} \n'.format(len(val_data)))
    print('Shape of input and output: ({},{}) \n'.format(x_train.shape[-2],x_train.shape[-1]))

    epochs = args.epochs
    lr = args.lr
    model_name = args.model_name
    continue_train = args.continue_train == 'True'

    optimizer = optim.Adam(model.parameters(), lr=lr)

    if os.path.exists("Metrics") == False:
        os.makedirs("Metrics")

    if not continue_train:
        # TRAINING CELL
        train_loss, val_loss = [], []
        train_psnr, val_psnr = [], []
        train_ssim, val_ssim = [], []
        start = time.time()

        last_epoch = -1
        vloss = np.inf

    else:
        # Load the lists of last time training metrics
        metrics = np.load(os.path.join("./Metrics",model_name + ".npy"))
        train_loss, val_loss = metrics[0].tolist(), metrics[3].tolist()
        train_psnr, val_psnr = metrics[1].tolist(), metrics[4].tolist()
        train_ssim, val_ssim = metrics[2].tolist(), metrics[5].tolist()
        start = time.time()

        # Model loading
        checkpoint = torch.load(model_name)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        last_epoch = checkpoint['epoch']
        losses = checkpoint['losses']
        vloss = losses[3]

    for epoch in range(last_epoch+1,epochs):
        print(f"Epoch {epoch + 1} of {epochs}")
        train_epoch_loss, train_epoch_psnr, train_epoch_ssim = train(model, train_loader, optimizer, train_data, max_val)
        val_epoch_loss, val_epoch_psnr, val_epoch_ssim = validate(model, val_loader, epoch, val_data, max_val)
        print(f"Train loss: {train_epoch_loss:.6f}")
        print(f"Val loss: {val_epoch_loss:.6f}")
        train_loss.append(train_epoch_loss)
        train_psnr.append(train_epoch_psnr)
        train_ssim.append(train_epoch_ssim)
        val_loss.append(val_epoch_loss)
        val_psnr.append(val_epoch_psnr)
        val_ssim.append(val_epoch_ssim)
        if val_epoch_loss < vloss:
            print("Save model...")
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'losses': [train_epoch_loss, train_epoch_psnr, train_epoch_ssim,val_epoch_loss, val_epoch_psnr, val_epoch_ssim],
                }, model_name)
            losses_path = os.path.join("./Metrics",model_name)
            metrics = [train_loss,train_psnr,train_ssim,val_loss,val_psnr,val_ssim]
            np.save(losses_path,metrics)
            vloss = val_epoch_loss
    end = time.time()
    print(f"Finished training in: {((end-start)/60):.3f} minutes")

if __name__ == '__main__':
    main()

























