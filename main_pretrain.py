import argparse
import datetime
import numpy as np
import os
import torch
from torch import nn
from torch.utils.tensorboard import SummaryWriter

from pretrain_BHSD import data_utils, metrics, engine_pretrain, utils
from models.unet3d.model import *
from models.mednext.MedNextV1 import *

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', default=8, type=int,
                        help='Batch size per GPU (effective batch size is batch_size * accum_iter * # gpus')
    parser.add_argument('--epochs', default=100, type=int)
    parser.add_argument('--patience', default=10, type=int)

    # Model parameters
    parser.add_argument('--model', default='Unet3d', type=str, metavar='MODEL',
                        help='Name of model to train')
    parser.add_argument('--img_size', default=256, type=int,
                        help='images input size')
    parser.add_argument('--n_slice', default=22, type=int,
                        help='number of image slice')

    # Optimizer parameters
    parser.add_argument('--opt', default='adamw', type=str, help='optimization algorithm')
    parser.add_argument('--lr', type=float, default=1e-4, metavar='LR',
                        help='learning rate (absolute lr)')
    parser.add_argument('--decay', type=float, default=1e-5,
                        help='weight decay (default: 1e-5)')

    # Dataset parameters
    parser.add_argument('--log_dir', default='./trained_model', type=str,
                        help='path where to save, empty for no saving')

    # training parameters
    parser.add_argument('--seed', default=178, type=int)
    parser.add_argument('--device', default='cuda',
                        help='device to use for training / testing')
    parser.add_argument('--num_workers', default=6, type=int)

    args = parser.parse_args()

    utils.set_seed(args.seed)
    device = torch.device(args.device)
    current_time = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    log_dir = os.path.join(args.log_dir, current_time)

    best_valid_score = np.inf
    lastmodel = None

    os.makedirs(log_dir, exist_ok=True)
    utils.save_args(args, log_dir, os.path.basename(__file__))
    log_writer = SummaryWriter(log_dir=log_dir)

    train_loader, valid_loader = data_utils.get_loader(args)

    if args.model == 'MedNeXt':
        model = MedNeXt(
                        in_channels = 1,                          # input channels
                        n_channels = 32,                          # number of base channels
                        n_classes = 1,                            # number of classes
                        exp_r = 4,                                # Expansion ratio in Expansion Layer
                        kernel_size = 7,                          # Kernel Size in Depthwise Conv. Layer
                        enc_kernel_size = None,                   # (Separate) Kernel Size in Encoder
                        dec_kernel_size = None,                   # (Separate) Kernel Size in Decoder
                        deep_supervision = False,                 # Enable Deep Supervision
                        do_res = False,                           # Residual connection in MedNeXt block
                        do_res_up_down = False,                   # Residual conn. in Resampling blocks
                        checkpoint_style = None,                  # Enable Gradient Checkpointing
                        block_counts = [1,1,1,1,2,1,1,1,1],       # Depth-first no. of blocks per layer 
                        norm_type = 'group',                      # Type of Norm: 'group' or 'layer'
                        dim = '3d'                                # Supports `3d', '2d' arguments
            )
    elif args.model == 'Unet3d':
        model = UNet3D(in_channels=1, out_channels=1, is_segmentation=False)
    model.to(device, non_blocking=True)

    criterion = metrics.DiceLoss()

    if args.opt == 'adam':
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.decay)
    elif args.opt == 'adamw':
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.decay)

    for epoch in range(1, args.epochs + 1):
        utils.info_message("EPOCH: {}", epoch)

        train_dic, valid_dic = engine_pretrain.train_one_epoch(
                    model, train_loader, valid_loader,
                    optimizer, criterion, epoch,
                    device, args=args
                    )
        
        # Train log_writer
        log_writer.add_scalar('Train/Loss', scalar_value=train_dic['loss'], global_step=epoch)
        log_writer.add_scalar('Train/Dice', scalar_value=train_dic['dice'], global_step=epoch)
        log_writer.add_scalar('Train/MSE', scalar_value=train_dic['mse'], global_step=epoch)
        # Valid log_writer
        log_writer.add_scalar('Train/Loss', scalar_value=valid_dic['loss'], global_step=epoch)
        log_writer.add_scalar('Train/Dice', scalar_value=valid_dic['dice'], global_step=epoch)
        log_writer.add_scalar('Train/MSE', scalar_value=valid_dic['mse'], global_step=epoch)

        utils.info_message(
            "[Epoch Train: {}] loss: {:.4f}, dice: {:.4f}, time: {:.2f} s",
            epoch, train_dic['loss'], train_dic['dice'], train_dic['time']
        )
        utils.info_message(
            "[Epoch Valid: {}] loss: {:.4f}, dice: {:.4f}, time: {:.2f} s",
            epoch, valid_dic['loss'], valid_dic['dice'], valid_dic['time']
        )

        if best_valid_score > valid_dic['loss']: 
            lastmodel = engine_pretrain.save_model(epoch, model, optimizer, best_valid_score, log_dir, args)
            utils.info_message(
                "auc improved from {:.4f} to {:.4f}. Saved model to '{}'", 
                best_valid_score, valid_dic['loss'], lastmodel
            )
            best_valid_score = valid_dic['loss']
            n_patience = 0
        else:
            n_patience += 1
            
        if n_patience >= args.patience:
            utils.info_message("\nValid auc didn't improve last {} epochs.", args.patience)
            break

if __name__ == "__main__":
    main()
