import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F

from val import matrix, calibration_metrics

def prob_dist(output, logits=True):
    if logits:
        probabilities = F.softmax(output, dim=1)
    else:
        probabilities = output
    confidence, i = torch.max(probabilities, dim=1)

    fig = plt.figure()
    ax = fig.add_subplot(1,1,1)
    ax.hist(confidence, bins=30)
    plt.show()

def calibration_plot(y_pred, y_label, n_bins=15):
    bin_boundaries = torch.linspace(0,1,n_bins+1)
    bin_lowers = bin_boundaries[:-1]
    bin_uppers = bin_boundaries[1:]

    confidences, predictions = torch.max(y_pred, 1)
    accuracies = predictions.eq(y_label)

    ece = torch.zeros(1)
    each_accuracy = []
    each_confidence = []
    weight = []

    for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
            # Calculated |confidence - accuracy| in each bin
            in_bin = confidences.gt(bin_lower.item()) * confidences.le(bin_upper.item())
            prop_in_bin = in_bin.float().mean()
            weight.append(prop_in_bin)
            if prop_in_bin > 0:
                    accuracy_in_bin = accuracies[in_bin].float().mean()
                    avg_confidence_in_bin = confidences[in_bin].mean()
                    each_accuracy.append(accuracy_in_bin)
                    each_confidence.append(avg_confidence_in_bin)

    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(1, 1, 1)
    ax.plot(each_confidence, each_accuracy, marker='o', color='blue', linestyle='-', linewidth=2)
    ax.plot([0, 1], [0, 1], color='red', linestyle='--', linewidth=1)

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.tick_params(labelsize=10)
    ax.grid(True, linestyle='--', linewidth=0.5)
    plt.show()

class ReliabilityDiagram(matrix.MaxProbCELoss):

    def plot(self, output, labels, n_bins = 15, logits = True, title = None):
        super().loss(output, labels, n_bins, logits)

        #computations
        delta = 1.0/n_bins
        x = torch.arange(0,1,delta)
        mid = torch.linspace(delta/2,1-delta/2,n_bins)
        error = torch.abs(mid - self.bin_acc)

        plt.rcParams["font.family"] = "serif"
        #size and axis limits
        plt.figure(figsize=(3,3))
        plt.xlim(0,1)
        plt.ylim(0,1)
        #plot grid
        plt.grid(color='tab:grey', linestyle=(0, (1, 5)), linewidth=1,zorder=0)
        #plot bars and identity line
        plt.bar(x, self.bin_acc, color = 'b', width=delta,align='edge',edgecolor = 'k',label='Outputs',zorder=5)
        plt.bar(x, error, bottom=torch.minimum(self.bin_acc,mid), color = 'mistyrose', alpha=0.5, width=delta,align='edge',edgecolor = 'r',hatch='/',label='Gap',zorder=10)
        ident = [0.0, 1.0]
        plt.plot(ident,ident,linestyle='--',color='tab:grey',zorder=15)
        #labels and legend
        plt.ylabel('Accuracy',fontsize=13)
        plt.xlabel('Confidence',fontsize=13)
        plt.legend(loc='upper left',framealpha=1.0,fontsize='medium')
        if title is not None:
            plt.title(title,fontsize=16)
        plt.tight_layout()
        plt.savefig("./ReliabilityDiagram.png", format="png")

        return plt

# class ReliabilityDiagram(calibration_metrics.CalibrationMetric):

#     def plot_bin(self, fx, y, title = None):
#         super().compute_error(fx, y)

#         delta = 1.0/self.num_bins
#         x = np.arange(0,1,delta)
#         mid = self.bin_conf
#         error = np.abs(mid - self.bin_acc)

#         #computations
#         # x = np.append(0, self.conf[:-1])
#         # x = np.append(x, 1)
#         # mid = self.bin_conf
#         # error = np.abs(self.bin_conf - self.bin_acc)

#         plt.rcParams["font.family"] = "serif"
#         #size and axis limits
#         plt.figure(figsize=(3,3))
#         plt.xlim(0,1)
#         plt.ylim(0,1)
#         #plot grid
#         plt.grid(color='tab:grey', linestyle=(0, (1, 5)), linewidth=1,zorder=0)
#         #plot bars and identity line
#         plt.bar(x, self.bin_acc, color = 'b', width=delta,align='edge',edgecolor = 'k',label='Confidence',zorder=5)
#         plt.bar(x, error, bottom=np.minimum(self.bin_acc,mid), color = 'mistyrose', alpha=0.5, width=delta,align='edge',edgecolor = 'r',hatch='/',label='CE',zorder=10)
#         ident = [0.0, 1.0]
#         plt.plot(ident,ident,linestyle='--',color='tab:grey',zorder=15)
#         #labels and legend
#         plt.ylabel('Accuracy',fontsize=13)
#         plt.xlabel('Confidence',fontsize=13)
#         plt.legend(loc='upper left',framealpha=1.0,fontsize='medium')
#         if title is not None:
#             plt.title(title,fontsize=16)
#         plt.tight_layout()
#         plt.show()
#         # plt.savefig("./Fig/ResNet_DE001_ew_ece_bin_L1.png", format="png")
#         # plt.savefig("./Fig/MLP-Mixer_DE001_ew_ece_bin_L1.png", format="png")
#         # plt.savefig("./Fig/Swin-Transformer_DE001_ew_ece_bin_L1.png", format="png")

#         return plt
    
#     def plot_sweep(self, fx, y, title = None):
#         super().compute_error(fx, y)

#         # bin_max = self.bin_max

#         #computations
#         x = np.append(0, self.bin_conf[:-1])
#         delta = np.append(x, 1)
#         delta = np.diff(delta)
#         mid = self.bin_conf
#         error = np.abs(self.bin_conf - self.bin_acc)

#         plt.rcParams["font.family"] = "serif"
#         #size and axis limits
#         plt.figure(figsize=(3,3))
#         plt.xlim(0,1)
#         plt.ylim(0,1)
#         #plot grid
#         plt.grid(color='tab:grey', linestyle=(0, (1, 5)), linewidth=1,zorder=0)
#         #plot bars and identity line
#         plt.bar(x, self.bin_acc, color = 'b', width=delta,align='edge',edgecolor = 'k',label='Confidence',zorder=5)
#         # plt.bar(x, error, bottom=torch.minimum(self.bin_acc,mid), color = 'mistyrose', alpha=0.5, width=delta,align='edge',edgecolor = 'r',hatch='/',label='Gap',zorder=10)
#         for i in range(len(x)):
#             plt.bar(x[i], error[i], bottom=np.minimum(self.bin_acc[i], mid[i]), color='mistyrose',
#                     alpha=0.5, width=delta[i], align='edge', edgecolor='r', hatch='/',
#                     label='CE' if i == 0 else '', zorder=10)
#         ident = [0.0, 1.0]
#         plt.plot(ident,ident,linestyle='--',color='tab:grey',zorder=15)
#         #labels and legend
#         plt.ylabel('Accuracy',fontsize=13)
#         plt.xlabel('Confidence',fontsize=13)
#         plt.legend(loc='upper left',framealpha=1.0,fontsize='medium')
#         if title is not None:
#             plt.title(title,fontsize=16)
#         plt.tight_layout()
#         plt.savefig("./Fig/MLP-Mixer_DE000_em_ece_sweep_L1.png", format="png")

#         return plt