
import torch
from utils import *
import os
from torchvision import transforms


class NDVI_LST(torch.utils.data.Dataset):
    """
    NDVI_LST dataset.
    refer to: https://pytorch.org/tutorials/beginner/data_loading_tutorial.html
    """
    def __init__(self, LST_1km_dir, LST_2km_dir, NDVI_1km_dir, NDVI_2km_dir, LST_1km_files, LST_2km_files, NDVI_1km_files, NDVI_2km_files, LST_to_NDVI_dic,\
                    root_dir='/content/drive/MyDrive/Ganglin/1_IMT/MCE_Projet3A/MODIS/', use_small=1,transform=None,mean_all=None, std_all=None):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
            root_dir = '/content/drive/MyDrive/Ganglin/1_IMT/MCE_Projet3A/MODIS/'
        """
        
        # self.root_dir = root_dir
        # self.transform = transform
        # self.LST_1km_dir = os.path.join(root_dir,"MOD_2020_MOD11A1/tifs_files/1km")
        # self.LST_2km_dir = os.path.join(root_dir,"MOD_2020_MOD11A1/tifs_files/2km")
        # self.NDVI_1km_dir = os.path.join(root_dir,"MOD_2020_MOD13A2/tifs_files/1km")
        # self.NDVI_2km_dir = os.path.join(root_dir,"MOD_2020_MOD13A2/tifs_files/2km")
        # self.LST_1km_files = os.listdir(self.LST_1km_dir)
        # self.LST_2km_files = os.listdir(self.LST_2km_dir)
        # self.NDVI_1km_files = os.listdir(self.NDVI_1km_dir)
        # self.NDVI_2km_files = os.listdir(self.NDVI_2km_dir)
        # self.LST_to_NDVI_dic = np.load(os.path.join(root_dir,"LST_to_NDVI_dic.npy"), allow_pickle=1).item()

        self.LST_1km_dir = LST_1km_dir
        self.LST_2km_dir = LST_2km_dir
        self.NDVI_1km_dir = NDVI_1km_dir
        self.NDVI_2km_dir = NDVI_2km_dir
        self.LST_1km_files = LST_1km_files
        self.LST_2km_files = LST_2km_files
        self.NDVI_1km_files = NDVI_1km_files
        self.NDVI_2km_files = NDVI_2km_files
        self.LST_to_NDVI_dic = LST_to_NDVI_dic
        self.transform = transform
        if use_small:
            self.LST_1km_files = self.LST_1km_files[:160]

        if self.transform:
            self.transform_ndvi_1km = transforms.Compose([transforms.ToTensor(),
                                                        transforms.Normalize(mean=mean_all["ndvi_1km"], std=std_all["ndvi_1km"])])
            self.transform_ndvi_2km = transforms.Compose([transforms.ToTensor(),
                                                        transforms.Normalize(mean=mean_all["ndvi_2km"], std=std_all["ndvi_2km"])])
            self.transform_LSTd_1km = transforms.Compose([transforms.ToTensor(),
                                                            transforms.Normalize(mean=mean_all["LSTd_1km"], std=std_all["LSTd_1km"])])
            self.transform_LSTd_2km = transforms.Compose([transforms.ToTensor(),
                                                            transforms.Normalize(mean=mean_all["LSTd_2km"], std=std_all["LSTd_2km"])])

    def __len__(self):
        return len(self.LST_1km_files)

    def __getitem__(self, idx):
        # get lst files
        lst_file_name = self.LST_1km_files[idx]
        name_list = lst_file_name.split(".")
        date = int(name_list[1].strip("A")[-4:])
        patch_num = name_list[-2]

        # get lst files
        tifs_1km_path = os.path.join(self.LST_1km_dir, lst_file_name)
        tifs_2km_path = os.path.join(self.LST_2km_dir, lst_file_name)
        LSTd_1km, LST_K_night_1km, cols_1km, rows_1km, projection_1km, geotransform_1km = read_tif(tifs_1km_path)
        LSTd_2km, LST_K_night_2km, cols_2km, rows_2km, projection_2km, geotransform_2km = read_tif(tifs_2km_path)

        # get ndvi files
        ndvi_file_name = self.LST_to_NDVI_dic[lst_file_name]
        ndvi_1km_path = os.path.join(self.NDVI_1km_dir, ndvi_file_name)
        ndvi_2km_path = os.path.join(self.NDVI_2km_dir, ndvi_file_name)
        ndvi_1km, ndvi_2km = self.get_ndvi(ndvi_1km_path, ndvi_2km_path)

        if self.transform:
            ndvi_1km = self.transform_ndvi_1km(ndvi_1km).squeeze(0)
            ndvi_2km = self.transform_ndvi_2km(ndvi_2km).squeeze(0)
            LSTd_1km = self.transform_LSTd_1km(LSTd_1km).squeeze(0)
            LSTd_2km = self.transform_LSTd_2km(LSTd_2km).squeeze(0)

        return ndvi_1km, LSTd_1km, ndvi_2km, LSTd_2km

    def get_ndvi(self, ndvi_1km_path, ndvi_2km_path):
        red, nir, mir, cols, rows, projection, geotransform = read_tif_MOD13A2(ndvi_1km_path)
        ndbi = (mir - nir) / (mir + nir)
        ndvi = (nir - red) / (nir + red)
        nan_index = np.isnan(ndvi)
        ndvi[nan_index] = 0
        ndvi_1km = ndvi.astype(np.float32)
        nan_index = np.isnan(ndbi)
        ndbi[nan_index] = 0
        ndbi_1km = ndbi.astype(np.float32)

        red, nir, mir, cols, rows, projection, geotransform = read_tif_MOD13A2(ndvi_2km_path)
        ndbi = (mir - nir) / (mir + nir)
        ndvi = (nir - red) / (nir + red)
        nan_index = np.isnan(ndvi)
        ndvi[nan_index] = 0
        ndvi_2km = ndvi.astype(np.float32)
        nan_index = np.isnan(ndbi)
        ndbi[nan_index] = 0
        ndbi_2km = ndbi.astype(np.float32)
        
        
        return ndvi_1km, ndvi_2km