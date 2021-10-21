# BulidDatabase
Hello guys! For the temporary code/notebook, I hope you can put them in **Dirty_code** folder, and for the **cleaned up** code, you can put it in the root folder directly.

## Metrics
https://cvnote.ddlee.cc/2019/09/12/psnr-ssim-python

## SRCNN
This is the simplest model for super resolution with only 3 layers. The model was trained with 100 epochs and aimed for a scale X2. The inputs have sizes of 256x256, then they are upscaled using bicubic to 512x512 and then fetched into the model (This will be removed in the future)

## SRCNN_64
Use the same network as SRCNN but for the new dataset 64x64 image size.
