import argparse
import datetime
import os
import torch
from torch import nn
from torch.utils.tensorboard import SummaryWriter

from train_Saiseikai import data_utils, extract_encoder, engine_train, utils, tensorboard_to_csv

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', default=8, type=int,
                        help='Batch size per GPU (effective batch size is batch_size * accum_iter * # gpus')
    parser.add_argument('--epochs', default=30, type=int)
    parser.add_argument('--patience', default=30, type=int)

    # Model parameters
    parser.add_argument('--model', default='Unet3d', type=str, metavar='MODEL',
                        help='Name of model to train')
    parser.add_argument('--img_size', default=256, type=int,
                        help='images input size')
    parser.add_argument('--n_slice', default=22, type=int,
                        help='number of image slice')

    # Optimizer parameters
    parser.add_argument('--opt', default='adamw', type=str, help='optimization algorithm')
    parser.add_argument('--lr', type=float, default=1e-5, metavar='LR',
                        help='learning rate (absolute lr)')
    parser.add_argument('--decay', type=float, default=1e-5,
                        help='weight decay (default: 1e-5)')

    # Dataset parameters
    parser.add_argument('--log_dir', default='./trained_model', type=str,
                        help='path where to save, empty for no saving')
    parser.add_argument('--event', default='mRS6', type=str,
                        help='event: mRS6, mRS3-5 or mRS3-6')
    parser.add_argument('--data_aug', default=5, type=int,
                        help='scalar multiplication of data volume')

    # Training parameters
    parser.add_argument('--seed', default=178, type=int)
    parser.add_argument('--device', default='cuda',
                        help='device to use for training / testing')
    parser.add_argument('--num_workers', default=8, type=int)
    parser.add_argument('--pos_weight', default=True, type=bool)
    parser.add_argument("--resume", default=None, type=str, help="resume training")

    args = parser.parse_args()

    utils.set_seed(args.seed)
    device = torch.device(args.device)
    current_time = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    log_dir = os.path.join(args.log_dir, current_time)

    best_valid_score = 0
    lastmodel = None

    os.makedirs(log_dir, exist_ok=True)
    utils.save_args(args, log_dir, os.path.basename(__file__))
    log_writer = SummaryWriter(log_dir=log_dir)

    train_loader, valid_loader, test_loader = data_utils.get_loader(args)

    if args.model == 'MedNeXt':
        model = extract_encoder.MedNeXt_en()
    elif args.model == 'Unet3d':
        if args.resume is not None:
            model_pth = args.resume
            model = extract_encoder.transfer_Unet3d_en(model_pth)
        else:
            model = extract_encoder.Unet3d_en()
    model.to(device, non_blocking=True)

    criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(data_utils.pos_weight(args)))

    if args.opt == 'adam':
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.decay)
    elif args.opt == 'adamw':
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.decay)

    for epoch in range(1, args.epochs + 1):
        utils.info_message("EPOCH: {}", epoch)

        train_dic, valid_dic, test_dic = engine_train.train_one_epoch(
                    model, train_loader, valid_loader, test_loader,
                    optimizer, criterion, epoch,
                    device, args=args
                    )
        
        # Train log_writer
        log_writer.add_scalar('Train/Loss', scalar_value=train_dic['loss'], global_step=epoch)
        log_writer.add_scalar('Train/ROC-AUC', scalar_value=train_dic['auc'], global_step=epoch)
        log_writer.add_scalar('Train/ACC', scalar_value=train_dic['acc'], global_step=epoch)
        log_writer.add_scalar('Train/F1', scalar_value=train_dic['f1'], global_step=epoch)
        log_writer.add_scalar('Train/SEN', scalar_value=train_dic['sen'], global_step=epoch)
        log_writer.add_scalar('Train/SPE', scalar_value=train_dic['spe'], global_step=epoch)
        log_writer.add_scalar('Train/PRE', scalar_value=train_dic['pre'], global_step=epoch)
        # Valid log_writer
        log_writer.add_scalar('Valid/Loss', scalar_value=valid_dic['loss'], global_step=epoch)
        log_writer.add_scalar('Valid/ROC-AUC', scalar_value=valid_dic['auc'], global_step=epoch)
        log_writer.add_scalar('Valid/ACC', scalar_value=valid_dic['acc'], global_step=epoch)
        log_writer.add_scalar('Valid/F1', scalar_value=valid_dic['f1'], global_step=epoch)
        log_writer.add_scalar('Valid/SEN', scalar_value=valid_dic['sen'], global_step=epoch)
        log_writer.add_scalar('Valid/SPE', scalar_value=valid_dic['spe'], global_step=epoch)
        log_writer.add_scalar('Valid/PRE', scalar_value=valid_dic['pre'], global_step=epoch)
        # Test log_writer
        log_writer.add_scalar('Test/Loss', scalar_value=test_dic['loss'], global_step=epoch)
        log_writer.add_scalar('Test/ROC-AUC', scalar_value=test_dic['auc'], global_step=epoch)
        log_writer.add_scalar('Test/ACC', scalar_value=test_dic['acc'], global_step=epoch)
        log_writer.add_scalar('Test/F1', scalar_value=test_dic['f1'], global_step=epoch)
        log_writer.add_scalar('Test/SEN', scalar_value=test_dic['sen'], global_step=epoch)
        log_writer.add_scalar('Test/SPE', scalar_value=test_dic['spe'], global_step=epoch)
        log_writer.add_scalar('Test/PRE', scalar_value=test_dic['pre'], global_step=epoch)

        utils.info_message(
            "[Epoch Train: {}] loss: {:.4f}, auc: {:.4f}, time: {:.2f} s",
            epoch, train_dic['loss'], train_dic['auc'], train_dic['time']
        )
        utils.info_message(
            "[Epoch Valid: {}] loss: {:.4f}, auc: {:.4f}, time: {:.2f} s",
            epoch, valid_dic['loss'], valid_dic['auc'], valid_dic['time']
        )
        utils.info_message(
            "[Epoch Test: {}] loss: {:.4f}, auc: {:.4f}, time: {:.2f} s",
            epoch, test_dic['loss'], test_dic['auc'], test_dic['time']
        )

        if best_valid_score < valid_dic['auc']: 
            lastmodel = engine_train.save_model(epoch, model, optimizer, best_valid_score, log_dir, args)
            utils.info_message(
                "auc improved from {:.4f} to {:.4f}. Saved model to '{}'", 
                best_valid_score, valid_dic['auc'], lastmodel
            )
            best_valid_score = valid_dic['auc']
            n_patience = 0
        else:
            n_patience += 1
            
        if n_patience >= args.patience:
            utils.info_message("\nValid auc didn't improve last {} epochs.", args.patience)
            break
    print('Done!!')
    utils.to_csv(log_dir)

if __name__ == "__main__":
    main()
