import argparse
import numpy as np
import os
import pandas as pd
from sklearn.model_selection import train_test_split
import torch

import data_utils, extract_encoder, utils

def to_csv(path):
    parser = argparse.ArgumentParser()
    # Model parameters
    parser.add_argument('--img_size', default=256, type=int,
                        help='images input size')
    parser.add_argument('--n_slice', default=22, type=int,
                        help='number of image slice')
    # Dataset parameters
    parser.add_argument('--event', default='mRS6', type=str,
                        help='event: mRS6 or mRS3-6')
    # Training parameters
    parser.add_argument('--device', default='cuda',
                            help='device to use for training / testing')
    args = parser.parse_args()

    device = torch.device(args.device)

    df = pd.read_csv('D:/main_saiseikai/v7_APAMI/poor_prognosis_add_headCTinfo.csv') # df.shape: (527, 8)

    if args.event == 'mRS6':
        df['obj'] = df['discharge_mRS'].apply(lambda x: 1 if x == 6 else 0) # event: mRS6, (1,0) = (76, 451)
    elif args.event == 'mRS3-6':
        df = df[df['discharge_mRS'] != 6] # df.shape: (451, 8)
        df['obj'] = df['discharge_mRS'].apply(lambda x: 1 if x >= 3 else 0) # event: mRS3-5, (1,0) = (344, 107)

    df = df[['pid', 'hospitalization_date', 'discharge_data', 'train_or_test', 'discharge_mRS', 'obj']]
    df_train, df_test = train_test_split(df, 
                                        test_size=0.4,
                                        stratify=df['obj'],
                                        random_state=22)
    train, valid = train_test_split(df_train,
                                        test_size=0.4,
                                        stratify=df_train['obj'],
                                        random_state=22)
    df_train = df_train.rename(columns={'train_or_test': 'train_or_valid'})
    df_train['train_or_valid'] = 'train'
    df_train.loc[df_train.index.isin(valid.index), 'train_or_valid'] = 'valid'
    df_train, df_test = df_train.reset_index(drop=True), df_test.reset_index(drop=True)

    train_dataset = data_utils.Dataset(df_train, df_train["pid"].values, args.n_slice, args.img_size)
    test_dataset = data_utils.Dataset(df_test, df_test["pid"].values, args.n_slice, args.img_size)
    
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=True)
    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True)

    model = extract_encoder.Unet3d_en()
    checkpoint = torch.load(path)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device, non_blocking=True)
    model.eval()

    # Train csv
    train_prob, train_label, train_latent = [], [], []

    for step, batch in enumerate(train_loader, 1):
        with torch.no_grad():
            X = batch["X"].to(device, non_blocking=True)
            targets = batch["y"].to(device, non_blocking=True)

            with torch.cuda.amp.autocast():
                outputs, latent = model(X)
                outputs = outputs.squeeze(1)
                prob = torch.sigmoid(outputs)
                train_prob.extend(prob.detach().clone().to('cpu').tolist())
                train_label.extend(batch["y"].detach().clone().to('cpu').tolist())
                train_latent.append(latent.detach().clone().to('cpu').tolist())

        message = 'Valid Step {}/{}'
        utils.info_message(message, step, len(train_loader), end="\r")

    df_prob = pd.DataFrame({'prob' : train_prob, 'label' : train_label})
    train_latent = torch.cat(train_latent, dim=0)
    df_latent = pd.DataFrame(train_latent.numpy(), columns=[f'dim{i}' for i in range(train_latent.shape[1])])
    df_con = pd.concat([df_train, df_prob, df_latent], axis=1)
    df_con.to_csv(os.path.join(path, 'train_prob_and_latent.csv'), index=False)

    # Test csv
    test_prob, test_label, test_latent = [], [], []
    
    for step, batch in enumerate(test_loader, 1):
        with torch.no_grad():
            X = batch["X"].to(device, non_blocking=True)
            targets = batch["y"].to(device, non_blocking=True)

            with torch.cuda.amp.autocast():
                outputs, latent = model(X)
                outputs = outputs.squeeze(1)
                prob = torch.sigmoid(outputs)
                test_prob.extend(prob.detach().clone().to('cpu').tolist())
                test_label.extend(batch["y"].detach().clone().to('cpu').tolist())
                test_latent.append(latent.detach().clone().to('cpu').tolist())

        message = 'Valid Step {}/{}'
        utils.info_message(message, step, len(test_loader), end="\r")

    df_prob = pd.DataFrame({'prob' : test_prob, 'label' : test_label})
    test_latent = torch.cat(test_latent, dim=0)
    df_latent = pd.DataFrame(test_latent.numpy(), columns=[f'dim{i}' for i in range(test_latent.shape[1])])
    df_con = pd.concat([df_test, df_prob, df_latent], axis=1)
    df_con.to_csv(os.path.join(path, 'test_prob_and_latent.csv'), index=False)

if __name__ == '__main__':
    path = './trained_model/20240617_190412'
    to_csv(path)
