# ichtrain

## Pretrain
```python
python main_pretrain.py
```

## Train
```python
python main_train.py --event 'mRS6' --resume './trained_model/20240626_142459/Unet3d.pth'
python main_train.py --event 'mRS3-6' --resume './trained_model/20240626_142459/Unet3d.pth'
python main_train.py --event 'mRS3-5' --resume './trained_model/20240626_142459/Unet3d.pth'
```
## Output results
```python
python main_to_csv.py --model_path './trained_model/20240702_155008/Unet3d.pth' --event 'mRS6'
python main_to_csv.py --model_path './trained_model/20240702_212307/Unet3d.pth' --event 'mRS3-6'
python main_to_csv.py --model_path './trained_model/20240702_230230/Unet3d.pth' --event 'mRS3-5'
```
