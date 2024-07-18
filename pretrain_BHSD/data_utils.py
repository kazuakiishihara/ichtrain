import numpy as np
import os
import random
import torch
from torch.utils import data as torch_data

from pretrain_BHSD import preprocessing

class Dataset(torch_data.Dataset):
    def __init__(self, data, args):
        self.data = data
        self.n_slice = args.n_slice
        self.img_size = args.img_size

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

    path_imgs = [os.path.join(folder_path_img, file_name) for file_name in file_names]
    path_labels = [os.path.join(folder_path_label, file_name) for file_name in file_names]

    li_img, li_label = [], []
    for path_img, path_label in zip(path_imgs, path_labels):
        mask = preprocessing.get_nifti(path_label)

        if np.sum(mask == 2) >= 1 or np.sum(mask == 3) >= 1:
            li_img.append(path_img)
            li_label.append(path_label)

    train_sampling = random.sample(range(len(li_img)), int(len(li_label)*0.7) )
    valid_sampling = [index for index in range(len(li_img)) if index not in train_sampling]

    train = [(li_img[index], li_label[index]) for index in train_sampling]
    valid = [(li_img[index], li_label[index]) for index in valid_sampling]

    train_dataset = Dataset(train, args)
    valid_dataset = Dataset(valid, args)
    
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=True)
    valid_loader = torch.utils.data.DataLoader(
        valid_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True)
    
    return train_loader, valid_loader
