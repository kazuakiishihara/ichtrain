import numpy as np
import pandas as pd
from PIL import Image
import random
from sklearn.model_selection import train_test_split
import torch
from torch.utils import data as torch_data
from torchvision import transforms as transforms

from train_Saiseikai import preprocessing

class Dataset(torch_data.Dataset):
    def __init__(self, csv, pid, n_slice, img_size):
        self.csv = csv
        self.pid = pid
        self.n_slice = n_slice
        self.img_size = img_size
    
    def __len__(self):
        return len(self.pid)
    
    def __getitem__(self, index):
        id = self.pid[index]
        filelist_str = self.csv.loc[self.csv["pid"] == id, "headCT_path_list"].values[0]
        img = preprocessing.load_dicom_image3d(filelist_str, self.n_slice, self.img_size)
        
        y = self.csv.loc[self.csv["pid"] == id, "obj"].values[0]
        return {"X" : torch.tensor(img).float(), "y" : torch.tensor(y).float()}

def aug(img):
    random_seed = random.randint(0,1e+8)
    transform = transforms.Compose([
        transforms.RandomRotation(degrees=30),
        transforms.RandomHorizontalFlip(0.3),
        transforms.RandomVerticalFlip(0.3)
    ])
    img_list = []
    for img_data in img:
        torch.manual_seed(random_seed)
        img_data = Image.fromarray(img_data)
        img_data = transform(img_data)
        img_data = np.array(img_data)
        img_data = img_data.tolist()
        img_list.append(img_data)
    
    return np.stack(img_list)

class Dataset_aug(torch_data.Dataset):
    def __init__(self, csv, pid, n_slice, img_size, args):
        self.csv = csv
        self.pid = [(el, 'none') for el in pid]
        time = args.data_aug
        for _ in range(time-1):
            self.pid.extend([(el, 'aug') for el in pid])
        self.n_slice = n_slice
        self.img_size = img_size

    def __len__(self):
        return len(self.pid)

    def __getitem__(self, index):
        id = self.pid[index][0]
        filelist_str = self.csv.loc[self.csv["pid"] == id, "headCT_path_list"].values[0]
        img = preprocessing.load_dicom_image3d(filelist_str, self.n_slice, self.img_size)
        y = self.csv.loc[self.csv["pid"] == id, "obj"].values[0]

        if self.pid[index][1] == 'aug':
            img = np.squeeze(img)
            img = aug(img)
            img = np.expand_dims(img, 0)
        
        return {"X" : torch.tensor(img).float(), "y" : torch.tensor(y).float()}

def split_data(args):
    df = pd.read_csv('D:/main_saiseikai/v7_APAMI/poor_prognosis_add_headCTinfo.csv') # df: (527, 8)

    if args.event == 'mRS6':
        df['obj'] = df['discharge_mRS'].apply(lambda x: 1 if x == 6 else 0) # event: mRS6, (1,0) = (76, 451)
    elif args.event == 'mRS3-6':
        df['obj'] = df['discharge_mRS'].apply(lambda x: 1 if x >= 3 and x <= 6 else 0) # event: mRS3-6, (1,0) = (420, 107)
    elif args.event == 'mRS3-5':
        df = df[df['discharge_mRS'] != 6] # df.shape: (451, 8)
        df['obj'] = df['discharge_mRS'].apply(lambda x: 1 if x >= 3 and x <= 5 else 0) # event: mRS3-5, (1,0) = (344, 107)
    
    df_train, df_test = df[df['train_or_test']=='train'], df[df['train_or_test']=='test'] 
    df_train, df_valid = train_test_split(df_train, test_size=0.3, stratify=df_train['obj'], random_state=22)

    return df_train, df_valid, df_test

def get_loader(args):
    df_train, df_valid, df_test = split_data(args)

    train_dataset = Dataset_aug(df_train, df_train["pid"].values, args.n_slice, args.img_size, args)
    valid_dataset = Dataset(df_valid, df_valid["pid"].values, args.n_slice, args.img_size)
    test_dataset = Dataset(df_test, df_test["pid"].values, args.n_slice, args.img_size)
    
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=True)
    valid_loader = torch.utils.data.DataLoader(
        valid_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True)
    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True)
    
    return train_loader, valid_loader, test_loader

def pos_weight(args):
    if args.pos_weight:
        df_train, df_valid, df_test = split_data(args)
        pos_weight = df_train['obj'].value_counts()[0] / df_train['obj'].value_counts()[1]
    else:
        pos_weight = 1
    return pos_weight
