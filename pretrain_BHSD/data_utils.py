import os
import random
import torch
from torch.utils import data as torch_data

from pretrain_BHSD import preprocessing

class Dataset(torch_data.Dataset):
    def __init__(self, data, n_slice=22, img_size=256):
        self.data = data # tapleのリスト
        self.n_slice = n_slice
        self.img_size = img_size

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, index):
        data_i = self.data[index]
        img, mask = preprocessing.load_image3d(data_i, self.n_slice, self.img_size)
        img, mask = torch.tensor(img).float(),  torch.tensor(mask).float() 
        return {"X" : img, "y" : mask}

def get_loader(args):
    folder_path_img = 'D:/Brain_Hemorrhage_Segmentation_Dataset/images'
    folder_path_label = 'D:/Brain_Hemorrhage_Segmentation_Dataset/ground_truths'

    file_names = os.listdir(folder_path_img)

    path_img = [os.path.join(folder_path_img, file_name) for file_name in file_names]
    path_label = [os.path.join(folder_path_label, file_name) for file_name in file_names]


    train_sampling = random.sample(range(192), 115)
    valid_sampling = [index for index in range(192) if index not in train_sampling]

    train = [(path_img[index], path_label[index]) for index in train_sampling]
    valid = [(path_img[index], path_label[index]) for index in valid_sampling]

    train_dataset = Dataset(train, args.n_slice, args.img_size)
    valid_dataset = Dataset(valid, args.n_slice, args.img_size)
    
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=True)
    valid_loader = torch.utils.data.DataLoader(
        valid_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True)
    
    return train_loader, valid_loader
