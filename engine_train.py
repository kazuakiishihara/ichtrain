import numpy as np
import os
import time
import torch

import utils

def train_one_epoch(model, train_loader, valid_loader, test_loader,
                    optimizer, criterion, epoch,
                    device, log_writer=None, args=None):
    train_dic = train_epoch(model, train_loader,
                optimizer, criterion, epoch,
                device, log_writer, args=None)
    valid_dic = valid_and_test_epoch(model, valid_loader,
                optimizer, criterion, epoch,
                device, args=None)
    test_dic = valid_and_test_epoch(model, test_loader,
                optimizer, criterion, epoch,
                device, args=None)
    return train_dic, valid_dic, test_dic

def train_epoch(model, loader,
                optimizer, criterion, epoch,
                device, log_writer=None, args=None):
    model.train()
    scaler = torch.cuda.amp.GradScaler()
    t = time.time()
    loss_train = []
    prob_li, label_li = [], []

    for step, batch in enumerate(loader, 1):
        X = batch["X"].to(device, non_blocking=True)
        targets = batch["y"].to(device, non_blocking=True)
        optimizer.zero_grad()

        with torch.cuda.amp.autocast():
            outputs = model(X)
            outputs = outputs.squeeze(1)
            prob = torch.sigmoid(outputs)
            prob_li.extend(prob.detach().clone().to('cpu').tolist())
            label_li.extend(batch["y"].detach().clone().to('cpu').tolist())
            loss = criterion(outputs, targets)
            loss_train.append(loss.detach().item())

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        message = 'Train Step {}/{}, train_loss: {:.4f}'
        utils.info_message(message, step, len(loader), np.mean(loss_train), end="\r")

    auc, acc, f1, sen, spe, pre = utils.metrics(label_li, prob_li)
    return {'loss' : np.mean(loss_train),
            'auc' : auc,
            'acc' : acc,
            'f1' : f1,
            'sen' : sen,
            'spe' : spe,
            'pre' : pre,
            'time' : int(time.time() - t)}

def valid_and_test_epoch(model, loader,
                optimizer, criterion, epoch,
                device, args=None):
    model.eval()
    t = time.time()
    loss_valid = []
    prob_li, label_li = [], []

    for step, batch in enumerate(loader, 1):
        with torch.no_grad():
            X = batch["X"].to(device, non_blocking=True)
            targets = batch["y"].to(device, non_blocking=True)

            with torch.cuda.amp.autocast():
                outputs = model(X)
                outputs = outputs.squeeze(1)
                prob = torch.sigmoid(outputs)
                prob_li.extend(prob.detach().clone().to('cpu').tolist())
                label_li.extend(batch["y"].detach().clone().to('cpu').tolist())
                loss = criterion(outputs, targets)
                loss_valid.append(loss.detach().item())

        message = 'Valid Step {}/{}, valid_loss: {:.4f}'
        utils.info_message(message, step, len(loader), np.mean(loss_valid), end="\r")

    auc, acc, f1, sen, spe, pre = utils.metrics(label_li, prob_li)
    return {'loss' : np.mean(loss_valid),
            'auc' : auc,
            'acc' : acc,
            'f1' : f1,
            'sen' : sen,
            'spe' : spe,
            'pre' : pre,
            'time' : int(time.time() - t)}

def save_model(epoch, model, optimizer, best_valid_score, output_dir, args):
    lastmodel =  os.path.join(output_dir, '{}.pth'.format(args.model))
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "best_valid_score": best_valid_score,
            "n_epoch": epoch,
        },
        lastmodel,
    )
    return lastmodel
