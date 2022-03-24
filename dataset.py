import torch
from torch.utils.data import Dataset

# THE DATASET CLASS
class LOADDataset(Dataset):
    def __init__(self, image_data, labels, transform=None):
        self.image_data = image_data
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return (len(self.image_data))
    def __getitem__(self, index):
        image = torch.tensor(self.image_data[index], dtype=torch.float)
        label = torch.tensor(self.labels[index], dtype=torch.float)
        if self.transform is not None:
          image = self.transform(image)
        
        return (image, label)