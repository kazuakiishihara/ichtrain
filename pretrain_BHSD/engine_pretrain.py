import os
import time
import torch

from pretrain_BHSD import metrics, utils

def train_one_epoch(model, train_loader, valid_loader,
                    optimizer, criterion, epoch,
                    device, args=None):
    train_dic = train_epoch(model, train_loader,
                optimizer, criterion, epoch,
                device, args=None)
    valid_dic = valid_and_test_epoch(model, valid_loader,
                optimizer, criterion, epoch,
                device, args=None)
    return train_dic, valid_dic

def train_epoch(model, loader,
                optimizer, criterion, epoch,
                device, args=None):
    model.train()
    scaler = torch.cuda.amp.GradScaler()
    t = time.time()
    sum_loss, sum_dice, sum_mse = 0, 0, 0

    for step, batch in enumerate(loader, 1):
        X = batch["X"].to(device, non_blocking=True)
        targets = batch["y"].to(device, non_blocking=True)
        optimizer.zero_grad()

        with torch.cuda.amp.autocast():
            outputs = model(X)
            loss = criterion(outputs, targets)
            sum_loss += loss.detach().item()
            dice = metrics.dice_coefficient(outputs, targets)
            sum_dice += dice.detach().item()
            mse = metrics.mse(outputs, targets)
            sum_mse += mse.detach().item()

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        message = 'Train Step {}/{}, train_loss: {:.4f}'
        utils.info_message(message, step, len(loader), sum_loss/step, end="\r")
    return {'loss' : sum_loss/len(loader),
            'dice' : sum_dice/len(loader),
            'mse' : sum_mse/len(loader),
            'time' : int(time.time() - t)}

def valid_and_test_epoch(model, loader,
                optimizer, criterion, epoch,
                device, args=None):
    model.eval()
    t = time.time()
    sum_loss, sum_dice, sum_mse = 0, 0, 0

    for step, batch in enumerate(loader, 1):
        with torch.no_grad():
            X = batch["X"].to(device, non_blocking=True)
            targets = batch["y"].to(device, non_blocking=True)

            with torch.cuda.amp.autocast():
                outputs = model(X)
                loss = criterion(outputs, targets)
                sum_loss += loss.detach().item()
                dice = metrics.dice_coefficient(outputs, targets)
                sum_dice += dice.detach().item()
                mse = metrics.mse(outputs, targets)
                sum_mse += mse.detach().item()

        message = 'Valid Step {}/{}, valid_loss: {:.4f}'
        utils.info_message(message, step, len(loader), sum_loss/step, end="\r")
    return {'loss' : sum_loss/len(loader),
            'dice' : sum_dice/len(loader),
            'mse' : sum_mse/len(loader),
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
