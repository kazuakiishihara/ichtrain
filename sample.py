import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#---------------------------------------------
# レーダー図
df = pd.read_csv('./trained_model/train_Unet3d mRS6/test.csv')
df_select = df.loc[8,:] # epoch: 9

df_fine = pd.read_csv('./trained_model/finetune_Unet3d/test.csv')
df_select_fine = df_fine.loc[7,:] # epoch: 8

# DataFrameから必要なデータを取得
labels = ['AUC', 'ACC', 'F1', 'Sensitivity', 'Specificity', 'Precision']  # 'Loss'を除外
stats_select = df_select[labels].values.tolist()
stats_select_fine = df_select_fine[labels].values.tolist()

labels = ['AUC', 'ACC', 'Sensitivity', 'F1', 'Specificity', 'Precision']
stats_select[2], stats_select[3] = stats_select[3], stats_select[2]
stats_select_fine[2], stats_select_fine[3] = stats_select_fine[3], stats_select_fine[2]

# データの数と角度を取得
num_vars = len(labels)
angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()

# 一周するように最初の軸を追加
stats_select += stats_select[:1]
stats_select_fine += stats_select_fine[:1]
angles += angles[:1]

custom_blue = (0.0, 0.298, 1.0)  # (R, G, B)
custom_red = (0.804, 0.361, 0.361)
# radar chartを描画
fig, ax = plt.subplots(figsize=(3.4, 3.4), subplot_kw=dict(polar=True))

ax.plot(angles, stats_select, color=custom_blue, linewidth=1, linestyle='solid', label='from scratch')  # df_selectのデータ
ax.fill(angles, stats_select, color=custom_blue, alpha=0.1)

ax.plot(angles, stats_select_fine, color=custom_red, linewidth=1, linestyle='solid', label='TL')  # df_select_fineのデータ
ax.fill(angles, stats_select_fine, color=custom_red, alpha=0.1)

# ax.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])  # 目盛りの位置を設定
# ax.set_yticklabels(['0.2', '0.4', '0.6', '0.8', '1.0'], fontsize=10)  # 目盛りのラベルを設定

# グラフの装飾
ax.set_yticklabels([])  # y軸のラベルを非表示に
ax.set_xticks(angles[:-1])
# ax.set_xticklabels(labels)
ax.set_xticklabels(labels, fontsize=10, fontname="Times New Roman", rotation=45)  

plt.legend(loc='upper right')
# plt.title('Radar Chart of Model Metrics')
plt.savefig('radar_chart.png', dpi=300)
plt.show()


#---------------------------------------------
import argparse

import data_utils, extract_encoder

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
parser.add_argument('--output_dir', default='./trained_model', type=str,
                    help='path where to save, empty for no saving')
parser.add_argument('--event', default='mRS6', type=str,
                    help='event: mRS6 or mRS3-5')

# training parameters
parser.add_argument('--seed', default=178, type=int)
parser.add_argument('--device', default='cuda',
                    help='device to use for training / testing')
parser.add_argument('--num_workers', default=8, type=int)
parser.add_argument('--pos_weight', default=5.75, type=float)
parser.add_argument("--resume", default=None, type=str, help="resume training")

args = parser.parse_args()
import torch
device = torch.device(args.device)
train_loader, valid_loader, test_loader = data_utils.get_loader(args)

def valid_and_test_epoch(model, loader,
                device, args=None):
    model.eval()
    logits, label_li = [], []
    for step, batch in enumerate(loader, 1):
        with torch.no_grad():
            X = batch["X"].to(device, non_blocking=True)
            targets = batch["y"].to(device, non_blocking=True)
            with torch.cuda.amp.autocast():
                outputs = model(X)
                outputs = outputs.squeeze(1)
                # prob = torch.sigmoid(outputs)
                logits.extend(outputs.detach().clone().to('cpu').tolist())
                label_li.extend(batch["y"].detach().clone().to('cpu').tolist())
    return torch.tensor(logits), torch.tensor(label_li)

model = extract_encoder.Unet3d_en()
# checkpoint = torch.load('./trained_model/train_Unet3d mRS6/Unet3d.pth')
checkpoint = torch.load('./trained_model/finetune_Unet3d/Unet3d.pth')
model.load_state_dict(checkpoint['model_state_dict'])
model.to(device, non_blocking=True)

logits, labels = valid_and_test_epoch(model, test_loader, device)
prob = torch.sigmoid(logits).numpy()
labels_np = labels.numpy()
cal_prob= prob[:, np.newaxis]
cal_label = labels_np[:, np.newaxis]


from val import calibration_metrics
fx, y = cal_prob, cal_label
cal = calibration_metrics.CalibrationMetric(
                                        ce_type="em_ece_bin",
                                        num_bins=15,
                                        bin_method="equal_width",
                                        norm=1,
                                        multiclass_setting="marginal")
cal.compute_error(cal_prob, cal_label)
binned_fx, binned_y, bin_sizes, bin_indices = cal._bin_data(fx, y)
calibration_error = cal._compute_error_all_binned(
                binned_fx, binned_y, bin_sizes)
ce = pow(np.abs(binned_fx - binned_y), 1) * bin_sizes

bin_conf = binned_fx.squeeze()
bin_acc = binned_y.squeeze()

import matplotlib.ticker as ticker


fig, ax = plt.subplots(figsize=(4, 4))
ax.plot(fx, y, 'x', label='Raw data', markersize=10)
n_bins = 15
bin_y = bin_acc
bin_centers = np.linspace(0, 1, n_bins, endpoint=False) + (1. / (n_bins * 2))
ax.bar(
        bin_centers,
        bin_y,
        width=1.0 / 15,
        color='r',
        alpha=0.3,
        linewidth=1,
        edgecolor='r',
        label='ECE Bins')
ax.plot([0.0, 1.0], [0.0, 1.0], 'k-', linewidth=2, label='True calibration')
ax.xaxis.set_major_locator(ticker.MultipleLocator(0.2))
ax.yaxis.set_major_locator(ticker.MultipleLocator(0.2))

# plt.title('Sample A')
# plt.xlabel('Predicted confidence')
# plt.ylabel('Empirical accuracy')
plt.yticks([])
ax.set_ylim([-0.05, 1.05])
ax.set_xlim([-0.05, 1.05])
ax.legend(loc=(0.02, 0.6), framealpha=0.4)
fig.savefig(
        'reliability_diagram_tl_Beta_alpha={}_beta={}.png'.format(2.8, 0.05),
        dpi='figure',
        bbox_inches='tight')

