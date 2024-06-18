# mRS6
# python main_train.py
# python main_train.py --resume './trained_model/pretrain_Unet3d/Unet3d.pth'

# mRS3-5
# python main_train.py --event 'mRS3-5'
# python main_train.py --event 'mRS3-5' --pos_weight 1
# python main_train.py --event 'mRS3-5' --pos_weight 0.311
# python main_train.py --event 'mRS3-5' --pos_weight 0.6
# python main_train.py --event 'mRS3-5' --pos_weight 1 --resume './trained_model/pretrain_Unet3d/Unet3d.pth'
# python main_train.py --event 'mRS3-5' --pos_weight 0.6 --resume './trained_model/pretrain_Unet3d/Unet3d.pth'

# latent
python main_train.py --event 'mRS6' --resume './trained_model/pretrain_Unet3d/Unet3d.pth'
# python main_train.py --event 'mRS3-6' --resume './trained_model/pretrain_Unet3d/Unet3d.pth'
