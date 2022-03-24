import torch
import cv2
import numpy as np
import argparse
import os
import matplotlib.pyplot as plt

from model import MRUNet
from tiff_process import tiff_process
from utils import downsampling

parser = argparse.ArgumentParser(description='PyTorch MR UNet inference on a folder of tif files.',
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--datapath',
                    help='path to the folder containing the tif files for inference')
parser.add_argument('--pretrained', help='path to pre-trained model')
parser.add_argument('--savepath', help='path to save figure')
parser.add_argument('--max_val', default=333.32000732421875, type=float, 
                    help='normalization factor for the input and output, which is the maximum pixel value of training data')
args = parser.parse_args()


def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Load pretrained model
    model_MRUnet = MRUNet(res_down=True, n_resblocks=1, bilinear=0).to(device)
    model_MRUnet.load_state_dict(torch.load(args.pretrained))
    # Normalization factor and scale
    max_val = args.max_val
    scale = 4


    # Tiff process
    Y_day, Y_night = tiff_process(args.datapath)

    # Test model
    indx = 0
    for y_day, y_night in zip(Y_day, Y_night):
        y_day_4km = downsampling(y_day, scale)
        y_night_4km = downsampling(y_night, scale)
        bicubic_day = cv2.resize(y_day_4km, y_day.shape, cv2.INTER_CUBIC)
        bicubic_night = cv2.resize(y_night_4km, y_night.shape, cv2.INTER_CUBIC)
        input_day = torch.tensor(np.reshape(bicubic_day/max_val, (1,1,64,64)), dtype=torch.float).to(device)
        input_night = torch.tensor(np.reshape(bicubic_night/max_val, (1,1,64,64)), dtype=torch.float).to(device)
        out_day = model_MRUnet(input_day).cpu().detach().numpy()*max_val
        out_night = model_MRUnet(input_night).cpu().detach().numpy()*max_val

        # plot and save fig   
        fontsize = 15
        clmap = 'jet'
        fig, ax = plt.subplots(nrows=2, ncols=2, figsize=(12, 10))
        im1 = ax[0,0].imshow(y_day, vmin = y_day.min(), vmax = y_day.max(), cmap = clmap)
        ax[0,0].set_title("Ground truth day", fontsize=fontsize)
        ax[0,0].axis('off')

        im2 = ax[0,1].imshow(out_day[0,0,:,:], vmin = y_day.min(), vmax = y_day.max(), cmap = clmap)
        ax[0,1].set_title("SR day", fontsize=fontsize)
        ax[0,1].axis('off')

        im3 = ax[1,0].imshow(y_night, vmin = y_night.min(), vmax = y_night.max(), cmap = clmap)
        ax[1,0].set_title("Ground truth night", fontsize=fontsize)
        ax[1,0].axis('off')

        im4 = ax[1,1].imshow(out_night[0,0,:,:], vmin = y_night.min(), vmax = y_night.max(), cmap = clmap)
        ax[1,1].set_title("SR night", fontsize=fontsize)
        ax[1,1].axis('off')

        # cmap = plt.get_cmap('jet',20)
        # fig.tight_layout()
        # fig.subplots_adjust(right=0.7)
        # cbar_ax = fig.add_axes([0.65, 0.15, 0.03, 0.7])
        # cbar_ax.tick_params(labelsize=15)
        # fig.colorbar(im1, cax=cbar_ax, cmap = clmap)

        plt.savefig(args.savepath + "/result_"+ str(indx)+ ".png",  bbox_inches='tight')
        indx += 1

if __name__ == '__main__':
    main()






