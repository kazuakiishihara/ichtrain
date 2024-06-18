"Refer to the following url"
"https://github.com/google-research/google-research/blob/master/uq_benchmark_2019/calibration_lib.py"

# how to use
# temperature = temperature_scaling_lib.find_scaling_temperature(logits, labels)
# scaled_probs = temperature_scaling_lib.apply_temperature_scaling(temperature, probs) # prob: Predictive probability of test set.
# temperatureパラメータは正の実数を取るが、関数では制約していないため注意！！

import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
from scipy.optimize import bisect

# def find_scaling_temperature(logits, labels, temp_range=(1e-5, 1e5)):
#     """Find max likelihood scaling temperature using binary search.
#     Arg:
#     labels, logits : Assume tensor type.
#     """
#     if len(labels.shape) != 1:
#         raise ValueError('Invalid labels shape=%s' % str(labels.shape))
#     if len(logits.shape) != 2:
#         raise ValueError('Invalid logits shape=%s' % str(logits.shape))
#     if len(labels) != logits.shape[0]:
#         raise ValueError('Incompatible shapes for logits (%s) vs labels (%s).' %
#                         (logits.shape, labels.shape))

#     def grad_fn(temperature):
#         """Returns gradient of log-likelihood WRT a logits-scaling temperature."""
#         temperature = torch.tensor(temperature, requires_grad=True)
#         if logits.dim() == 1:
#             probs = F.softmax(logits / temperature, dim=0)
#             nll = F.cross_entropy(logits, labels, reduction='sum')
#         elif logits.dim() == 2:
#             probs = F.softmax(logits / temperature, dim=1)
#             nll = F.nll_loss(torch.log(probs), labels, reduction='sum')
#         grad = torch.autograd.grad(nll, temperature)[0].item()
#         return grad

#     tmin, tmax = temp_range
#     temperature = bisect(grad_fn, tmin, tmax)
#     return temperature

def find_scaling_temperature(logits, labels, learning_rate=0.01, num_iterations=1000):
    temp = []
    temperature = torch.tensor(1.0, requires_grad=True)
    optimizer = torch.optim.SGD([temperature], lr=learning_rate)

    for _ in range(num_iterations):        
        optimizer.zero_grad()

        if logits.dim() == 1:
            probs = F.softmax(logits / temperature, dim=0)
            nll = F.cross_entropy(logits, labels, reduction='sum')
        elif logits.dim() == 2:
            probs = F.softmax(logits / temperature, dim=1)
            nll = F.nll_loss(torch.log(probs), labels, reduction='sum')
        
        nll.backward()
        optimizer.step()

        temp.append(temperature.item())
    plt.plot(range(len(temp)), temp)
    plt.show()

    return temperature.item()


def apply_temperature_scaling(temperature, probs):
    """Apply temperature scaling to an array of probabilities."""
    logits_t = torch.log(probs) / temperature
    scaled_probs = F.softmax(logits_t, dim=1)
    return scaled_probs