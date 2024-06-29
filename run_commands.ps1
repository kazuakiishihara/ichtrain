python main_pretrain.py
python main_train.py --event 'mRS6' --resume './trained_model/20240626_142459/Unet3d.pth'
python main_train.py --event 'mRS3-6' --resume './trained_model/20240626_142459/Unet3d.pth' --pos_weight 3
