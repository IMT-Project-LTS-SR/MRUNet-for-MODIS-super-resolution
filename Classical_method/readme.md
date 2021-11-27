# Requirements
conda install -c conda-forge gdal -y
pip install pymodis
pip install pymp-pypi
pip install opencv-python

# Download MODIS data automatically
nohup python -u modis_downloader.py --year_begin 2000 --year_end 2010 > modis_downloader_2001_2010 2>&1 &

nohup python -u modis_downloader.py --year_begin 2000 --year_end 2001 > modis_downloader2001 2>&1 &
nohup python -u modis_downloader.py --year_begin 2001 --year_end 2002 > modis_downloader2002 2>&1 &
nohup python -u modis_downloader.py --year_begin 2002 --year_end 2003 > modis_downloader2003 2>&1 &
nohup python -u modis_downloader.py --year_begin 2003 --year_end 2004 > modis_downloader2004 2>&1 &
nohup python -u modis_downloader.py --year_begin 2005 --year_end 2006 > modis_downloader2006 2>&1 &
nohup python -u modis_downloader.py --year_begin 2006 --year_end 2007 > modis_downloader2007 2>&1 &
nohup python -u modis_downloader.py --year_begin 2007 --year_end 2008 > modis_downloader2008 2>&1 &
nohup python -u modis_downloader.py --year_begin 2008 --year_end 2009 > modis_downloader2009 2>&1 &
nohup python -u modis_downloader.py --year_begin 2009 --year_end 2010 > modis_do/opt/img/MATLAB_R2021a/bin/matlabdownloader2017 2>&1 &
nohup python -u modis_downloader.py --year_begin 2017 --year_end 2018 > modis_downloader2018 2>&1 &
nohup python -u modis_downloader.py --year_begin 2018 --year_end 2019 > modis_downloader2019 2>&1 &
nohup python -u modis_downloader.py --year_begin 2019 --year_end 2020 > modis_downloader2020 2>&1 &
nohup python -u modis_downloader.py --year_begin 2020 --year_end 2021 > modis_downloader2021 2>&1 &

nohup python -u modis_data_preprocessing.py --year_begin 2000 --year_end 2001 > modis_data_preprocessing2001 2>&1 &
nohup python -u modis_data_preprocessing.py --year_begin 2001 --year_end 2002 > modis_data_preprocessing2002 2>&1 &
nohup python -u modis_data_preprocessing.py --year_begin 2002 --year_end 2003 > modis_data_preprocessing2003 2>&1 &
nohup python -u modis_data_preprocessing.py --year_begin 2003 --year_end 2004 > modis_data_preprocessing2004 2>&1 &
nohup python -u modis_data_preprocessing.py --year_begin 2005 --year_end 2006 > modis_data_preprocessing2006 2>&1 &
nohup python -u modis_data_preprocessing.py --year_begin 2006 --year_end 2007 > modis_data_preprocessing2007 2>&1 &
nohup python -u modis_data_preprocessing.py --year_begin 2007 --year_end 2008 > modis_data_preprocessing2008 2>&1 &
nohup python -u modis_data_preprocessing.py --year_begin 2008 --year_end 2009 > modis_data_preprocessing2009 2>&1 &
nohup python -u modis_data_preprocessing.py --year_begin 2009 --year_end 2010 > modis_data_preprocessing2010 2>&1 &
nohup python -u modis_data_preprocessing.py --year_begin 2010 --year_end 2011 > modis_data_preprocessing2011 2>&1 &
nohup python -u modis_data_preprocessing.py --year_begin 2011 --year_end 2012 > modis_data_preprocessing2012 2>&1 &
nohup python -u modis_data_preprocessing.py --year_begin 2012 --year_end 2013 > modis_data_preprocessing2013 2>&1 &
nohup python -u modis_data_preprocessing.py --year_begin 2013 --year_end 2014 > modis_data_preprocessing2014 2>&1 &
nohup python -u modis_data_preprocessing.py --year_begin 2014 --year_end 2015 > modis_data_preprocessing2015 2>&1 &
nohup python -u modis_data_preprocessing.py --year_begin 2015 --year_end 2016 > modis_data_preprocessing2016 2>&1 &
nohup python -u modis_data_preprocessing.py --year_begin 2016 --year_end 2017 > modis_data_preprocessing2017 2>&1 &
nohup python -u modis_data_preprocessing.py --year_begin 2017 --year_end 2018 > modis_data_preprocessing2018 2>&1 &
nohup python -u modis_data_preprocessing.py --year_begin 2018 --year_end 2019 > modis_data_preprocessing2019 2>&1 &
nohup python -u modis_data_preprocessing.py --year_begin 2019 --year_end 2020 > modis_data_preprocessing2020 2>&1 &
nohup python -u modis_data_preprocessing.py --year_begin 2020 --year_end 2021 > modis_data_preprocessing2021 2>&1 &

<!-- tail -f modis_downloader -->

tensorboard --logdir=outputs/UNet_Channel_Fusion_Cubic_Inputs_Add/logevs

# Pre-process the data (eliminate the cloud and sea pixel)
nohup python -u Downloader/modis_data_preprocessing.py > modis_preprocessor 2>&1 &
tail -f modis_preprocessor

nohup python -u modis_downloader.py --year_begin 2002 --year_end 2004 > modis_downloader 2>&1 &
nohup python -u modis_data_preprocessing.py > modis_data_preprocessing 2>&1 &

# Model test:
The simple lay concatenation doesn't generate better performance than Bicubic.

# Note
/opt/img/MATLAB_R2021a/bin/matlab -nodesktop -nosplash
/opt/img/MATLAB_R2018b/bin/matlab -nodesktop -nosplash

# Dataset transfer
scp -v -r g19tian@srv-df-895:/users/local/Downloader/MODIS_just_tif.zip /users/local/Dataset/
scp -v -r g19tian@srv-df-895:/users/local/Downloader/MODIS_just_tif.zip g19tian@srv-df-893:/users/local/Dataset/
scp -v -r g19tian@srv-df-895:/users/local/Downloader/MODIS_just_tif.zip g19tian@srv-df-894:/users/local/Dataset/
scp -v -r g19tian@srv-df-895:/users/local/Downloader/MODIS_just_tif.zip g19tian@srv-df-896:/users/local/Dataset/

scp -v -r /users/local/MODIS g19tian@srv-df-892:/users/local/MODIS


scp -v -r C:/Users/11849/Downloads/MODIS_just_tif.zip g19tian@srv-df-892:/users/local/Dataset/
scp -v -r C:/Users/11849/Downloads/MODIS_just_tif.zip g19tian@srv-df-893:/users/local/Dataset/
scp -v -r C:/Users/11849/Downloads/MODIS_just_tif.zip g19tian@srv-df-894:/users/local/Dataset/
scp -v -r C:/Users/11849/Downloads/MODIS_just_tif.zip g19tian@srv-df-896:/users/local/Dataset/

# test code 

nohup python -u train.py UNet_Minh > ~/Documents/0_test/MSE_Grad_loss/UNet_Minh 2>&1 &
nohup python -u train.py UNet_Channel_Fusion_Cubic_Inputs > ~/Documents/0_test/MSE_Grad_loss/UNet_Channel_Fusion_Cubic_Inputs 2>&1 &
nohup python -u train.py UNet_Channel_Fusion_Cubic_Inputs_Add > ~/Documents/0_test/MSE_Grad_loss/UNet_Channel_Fusion_Cubic_Inputs_Add 2>&1 &
nohup python -u train.py Encoder_Decoder > ~/Documents/0_test/MSE_Grad_loss/Encoder_Decoder 2>&1 &



# Vis

