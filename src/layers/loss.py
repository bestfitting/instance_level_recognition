import sys
sys.path.insert(0, '../..')
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from config.config import *

class LabelSmoothingLoss(nn.Module):
  def __init__(self, smoothing=0.1):
    super(LabelSmoothingLoss, self).__init__()
    self.confidence = 1.0 - smoothing
    self.smoothing = smoothing

  def forward(self, logits, labels, epoch=0, **kwargs):
    if self.training:
      logits = logits.float()
      labels = labels.float()
      logprobs = torch.nn.functional.log_softmax(logits, dim=-1)

      nll_loss = -logprobs.gather(dim=-1, index=labels.unsqueeze(1).long())
      nll_loss = nll_loss.squeeze(1)
      smooth_loss = -logprobs.mean(dim=-1)
      loss = self.confidence * nll_loss + self.smoothing * smooth_loss
      loss = loss.mean()
    else:
      loss = F.cross_entropy(logits, labels)
    return loss

class LabelSmoothingLossV1(nn.modules.Module):
  def __init__(self):
    super(LabelSmoothingLossV1, self).__init__()
    self.classify_loss = LabelSmoothingLoss()

  def forward(self, logits, labels, epoch=0):
    out_face, feature = logits
    loss = self.classify_loss(out_face, labels)
    return loss

if __name__ == "__main__":
  loss = LabelSmoothingLossV1()
  logits = Variable(torch.randn(3, NUM_CLASSES))
  labels = Variable(torch.LongTensor(3).random_(NUM_CLASSES))
  output = loss([logits, None, logits], labels)
  print(output)
