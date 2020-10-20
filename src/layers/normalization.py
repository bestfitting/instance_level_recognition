import torch
import torch.nn as nn

# --------------------------------------
# Normalization layers
# --------------------------------------
def l2n(x, eps=1e-6):
  return x / (torch.norm(x, p=2, dim=1, keepdim=True) + eps).expand_as(x)

class L2N(nn.Module):

  def __init__(self, eps=1e-6):
    super(L2N, self).__init__()
    self.eps = eps

  def forward(self, x):
    return l2n(x, eps=self.eps)

  def __repr__(self):
    return self.__class__.__name__ + '(' + 'eps=' + str(self.eps) + ')'
