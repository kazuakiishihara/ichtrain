import argparse
import numpy as np
import os
import pandas as pd
from sklearn.model_selection import train_test_split
import torch

from train_Saiseikai import data_utils, extract_encoder, utils

def to_csv():
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', default=8, type=int,
                        help='Batch size per GPU (effective batch size is batch_size * accum_iter * # gpus')
    # Model parameters
    parser.add_argument('--img_size', default=256, type=int,
                        help='images input size')
    parser.add_argument('--n_slice', default=22, type=int,
                        help='number of image slice')
    # Dataset parameters
    parser.add_argument('--log_dir', default='./trained_model/result', type=str,
                        help='path where to save, empty for no saving')
    parser.add_argument('--model_path', default='./trained_model/20240702_155008/Unet3d.pth', type=str,
                        help='trained model path')
    parser.add_argument('--event', default='mRS6', type=str,
                        help='event: mRS6 or mRS3-6')
    # Training parameters
    parser.add_argument('--device', default='cuda',
                            help='device to use for training / testing')
    parser.add_argument('--num_workers', default=8, type=int)
    args = parser.parse_args()

    device = torch.device(args.device)
    log_dir = args.log_dir
    os.makedirs(log_dir, exist_ok=True)

    df = pd.read_csv('D:/main_saiseikai/v8/poor_prognosis_add_headCTinfo.csv') # df.shape: (527, 8)

    if args.event == 'mRS6':
        df['obj'] = df['discharge_mRS'].apply(lambda x: 1 if x == 6 else 0) # event: mRS6, (1,0) = (76, 451)
    elif args.event == 'mRS3-6':
        df['obj'] = df['discharge_mRS'].apply(lambda x: 1 if x >= 3 and x <= 6 else 0) # event: mRS3-6, (1,0) = (420, 107)
    elif args.event == 'mRS3-5':
        df = df[df['discharge_mRS'] != 6] # df.shape: (451, 8)
        df['obj'] = df['discharge_mRS'].apply(lambda x: 1 if x >= 3 and x <= 5 else 0) # event: mRS3-5, (1,0) = (344, 107)
    
    df = df[['pid', 'hospitalization_date', 'discharge_data', 'train_or_test', 'discharge_mRS', 'obj', 'headCT_path_list']]
    df_train, df_test = df[df['train_or_test']=='train'], df[df['train_or_test']=='test'] 
    train, valid = train_test_split(df_train, test_size=0.3, stratify=df_train['obj'], random_state=22)

    df_train = df_train.rename(columns={'train_or_test': 'train_or_valid'})
    df_train['train_or_valid'] = 'train'
    df_train.loc[df_train.index.isin(valid.index), 'train_or_valid'] = 'valid'
    df_train, df_test = df_train.reset_index(drop=True), df_test.reset_index(drop=True)

    train_dataset = data_utils.Dataset(df_train, args)
    test_dataset = data_utils.Dataset(df_test, args)
    
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True)
    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True)

    model = extract_encoder.Unet3d_en()
    checkpoint = torch.load(args.model_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device, non_blocking=True)
    model.eval()

    # Train csv
    train_latent512, train_latent128 = [], []
    for step, batch in enumerate(train_loader, 1):
        with torch.no_grad():
            X = batch["X"].to(device, non_blocking=True)
            with torch.cuda.amp.autocast():
                outputs, latent512, latent128 = model(X)
                outputs = outputs.squeeze(1)
                train_latent512.append(latent512.detach().clone().to('cpu'))
                train_latent128.append(latent128.detach().clone().to('cpu'))

        message = 'Train Step {}/{}'
        utils.info_message(message, step, len(train_loader), end="\r")

    df_train = df_train.drop('headCT_path_list', axis=1)

    # Latent dimention: 512
    train_latent512 = torch.cat(train_latent512, dim=0)
    df_train_latent512 = pd.DataFrame(train_latent512.numpy(), columns=[f'dim{i}' for i in range(train_latent512.shape[1])])
    df_train_latent512 = pd.concat([df_train, df_train_latent512], axis=1)
    df_train_latent512.to_csv(os.path.join(log_dir, args.event + '_TrainValid_latent512.csv'), index=False)

    # Latent dimention: 128
    train_latent128 = torch.cat(train_latent128, dim=0)
    df_train_latent128 = pd.DataFrame(train_latent128.numpy(), columns=[f'dim{i}' for i in range(train_latent128.shape[1])])
    df_train_latent128 = pd.concat([df_train, df_train_latent128], axis=1)
    df_train_latent128.to_csv(os.path.join(log_dir, args.event + '_TrainValid_latent128.csv'), index=False)

    # Test csv
    test_prob = []
    test_latent512, test_latent128 = [], []
    for step, batch in enumerate(test_loader, 1):
        with torch.no_grad():
            X = batch["X"].to(device, non_blocking=True)
            with torch.cuda.amp.autocast():
                outputs, latent512, latent128 = model(X)
                outputs = outputs.squeeze(1)
                prob = torch.sigmoid(outputs)
                test_prob.extend(prob.detach().clone().to('cpu').tolist())
                test_latent512.append(latent512.detach().clone().to('cpu'))
                test_latent128.append(latent128.detach().clone().to('cpu'))

        message = 'Test Step {}/{}'
        utils.info_message(message, step, len(test_loader), end="\r")

    df_test = df_test.drop('headCT_path_list', axis=1)

    df_prob = pd.DataFrame({'prob' : test_prob})
    df_test_probLatent = pd.concat([df_test, df_prob], axis=1)
    df_test_probLatent.to_csv(os.path.join(log_dir, args.event + '_Test_prob.csv'), index=False)

    # Latent dimention: 512
    test_latent512 = torch.cat(test_latent512, dim=0)
    df_test_latent512 = pd.DataFrame(test_latent512.numpy(), columns=[f'dim{i}' for i in range(test_latent512.shape[1])])
    df_test_latent512 = pd.concat([df_test, df_test_latent512], axis=1)
    df_test_latent512.to_csv(os.path.join(log_dir, args.event + '_Test_latent512.csv'), index=False)

    # Latent dimention: 128
    test_latent128 = torch.cat(test_latent128, dim=0)
    df_test_latent128 = pd.DataFrame(test_latent128.numpy(), columns=[f'dim{i}' for i in range(test_latent128.shape[1])])
    df_test_latent128 = pd.concat([df_test, df_test_latent128], axis=1)
    df_test_latent128.to_csv(os.path.join(log_dir, args.event + '_Test_latent128.csv'), index=False)

if __name__ == '__main__':
    to_csv()
