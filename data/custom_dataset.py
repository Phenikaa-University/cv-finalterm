import os
from PIL import Image
import pandas as pd
from torch.utils.data import Dataset
from torchvision import transforms
import numpy as np

class CustomDataset(Dataset):
    def __init__(self, image_dir, labels_file, transform=None):
        self.image_dir = image_dir
        self.df = pd.read_excel(labels_file)
        self.image_name = self.df['img_url'].str.split('/').str[-1]
        self.labels = self.df['label_detail']
        self.transform = transform
    
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        I_path = os.path.join(self.image_dir, self.df.iloc[idx]['img_url'].split('/')[-1])
        image_name = self.df.iloc[idx]['img_url'].split('/')[-1]
        I = Image.open(I_path).convert('RGB')
        if I is None:
            raise FileNotFoundError(f'Image not found: img_name')
        
        label_str = self.df.iloc[idx]['label_detail']

        image = self.transform(I)
        label_str_set = ['0', '1', '2', '3', '4', '5', '6', '7']
        labels = convert_label(label_str, label_str_set)
        labels_array = np.zeros(8, dtype=np.float32)
        labels_array[labels] = 1.0
        return image, labels_array, image_name


def convert_label(label_str, label_str_set):
    labels = []
    for char in label_str:
        if char in label_str_set:
            labels.append(int(char))
    return labels