# from https://github.com/ronghuaiyang/arcface-pytorch/blob/master/models/metrics.py
# adacos: https://github.com/4uiiurz1/pytorch-adacos/blob/master/metrics.py
from __future__ import print_function
from __future__ import division
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.autograd
from torch.nn import Parameter
import math

class ArcMarginProduct(nn.Module):
  r"""Implement of large margin arc distance: :
    Args:
      in_features: size of each input sample
      out_features: size of each output sample
      s: norm of input feature
      m: margin
      cos(theta + m)
    """
  def __init__(self, in_features, out_features, s=30.0, m=0.50, easy_margin=False, ls_eps=0.0):
    super(ArcMarginProduct, self).__init__()
    self.in_features = in_features
    self.out_features = out_features
    self.s = s
    self.m = m
    self.ls_eps = ls_eps  # label smoothing
    self.W = Parameter(torch.FloatTensor(out_features, in_features))
    self.reset_parameters()

    self.easy_margin = easy_margin
    self.cos_m = math.cos(m)
    self.sin_m = math.sin(m)
    self.th = math.cos(math.pi - m)
    self.mm = math.sin(math.pi - m) * m

  def reset_parameters(self):
    stdv = 1. / math.sqrt(self.W.size(1))
    self.W.data.uniform_(-stdv, stdv)

  def forward(self, input, label):
    # --------------------------- cos(theta) & phi(theta) ---------------------------
    cosine = F.linear(F.normalize(input), F.normalize(self.W))
    if label is None:
      return cosine
    sine = torch.sqrt(1.0 - torch.pow(cosine, 2))
    phi = cosine * self.cos_m - sine * self.sin_m
    if self.easy_margin:
      phi = torch.where(cosine.float() > 0, phi, cosine.float())
    else:
      phi = torch.where(cosine.float() > self.th, phi, cosine.float() - self.mm)
    # --------------------------- convert label to one-hot ---------------------------
    # one_hot = torch.zeros(cosine.size(), requires_grad=True, device='cuda')
    one_hot = torch.zeros(cosine.size(), device=label.device)
    one_hot.scatter_(1, label.view(-1, 1).long(), 1)
    if self.ls_eps > 0:
      one_hot = (1 - self.ls_eps) * one_hot + self.ls_eps / self.out_features
    # -------------torch.where(out_i = {x_i if condition_i else y_i) -------------
    output = (one_hot * phi) + ((1.0 - one_hot) * cosine)
    output *= self.s

    return output

class ArcMarginProduct2(nn.Module):

  def __init__(self, in_features, out_features, s=30.0, m=0.50, easy_margin=False, ls_eps=0.0):
    super(ArcMarginProduct2, self).__init__()
    self.in_features = in_features
    self.out_features = out_features
    self.s = s
    self.m = m
    self.ls_eps = ls_eps  # label smoothing
    self.W = Parameter(torch.FloatTensor(out_features, in_features))
    nn.init.xavier_uniform_(self.W)

    self.easy_margin = easy_margin
    self.cos_m = math.cos(m)
    self.sin_m = math.sin(m)
    self.th = math.cos(math.pi - m)
    self.mm = math.sin(math.pi - m) * m

  def forward(self, input, label):
    # --------------------------- cos(theta) & phi(theta) ---------------------------
    cosine = F.linear(F.normalize(input), F.normalize(self.W))
    if label == None:
      return cosine
    sine = torch.sqrt(1.0 - torch.pow(cosine, 2))
    phi = cosine * self.cos_m - sine * self.sin_m
    if self.easy_margin:
      phi = torch.where(cosine.float() > 0, phi, cosine.float())
    else:
      phi = torch.where(cosine.float() > self.th, phi, cosine.float() - self.mm)
    # --------------------------- convert label to one-hot ---------------------------
    # one_hot = torch.zeros(cosine.size(), requires_grad=True, device='cuda')
    one_hot = torch.zeros(cosine.size(), device=label.device)
    one_hot.scatter_(1, label.view(-1, 1).long(), 1)
    if self.ls_eps > 0:
      one_hot = (1 - self.ls_eps) * one_hot + self.ls_eps / self.out_features
    # -------------torch.where(out_i = {x_i if condition_i else y_i) -------------
    output = (one_hot * phi) + ((1.0 - one_hot) * cosine)
    output *= self.s

    return output
