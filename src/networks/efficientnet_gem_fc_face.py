import sys
sys.path.insert(0, '..')

from layers.scheduler import *
from layers.loss import *
from config.config import *
from layers.pooling import gem
from layers.backbone.efficientnet_pytorch import EfficientNet
from layers.metric_learning import *

## net  ######################################################################
class ClsClass(nn.Module):

  def __init__(self,
               args,
               feature_net='efficientnet_b5',
               loss_module='AdaCos',
               margin=0.0,
               s=30.0,
               ):
    super().__init__()
    num_classes = args.num_classes

    if feature_net == 'efficientnet_b5':
      self.backbone = EfficientNet.from_pretrained('efficientnet-b5', model_dir=PRETRAINED_DIR, can_print=args.can_print)
      feat_dim = 2048
    elif feature_net == 'efficientnet_b6':
      self.backbone = EfficientNet.from_pretrained('efficientnet-b6', model_dir=PRETRAINED_DIR, can_print=args.can_print)
      feat_dim = 2304
    elif feature_net == 'efficientnet_b7':
      self.backbone = EfficientNet.from_pretrained('efficientnet-b7', model_dir=PRETRAINED_DIR, can_print=args.can_print)
      feat_dim = 2560

    self.in_channels = args.in_channels
    self.pool = gem
    fc_dim = 512
    self.fc = nn.Linear(feat_dim, fc_dim)
    self.bn = nn.BatchNorm1d(fc_dim)
    if loss_module == 'arcface':
      self.face_margin_product = ArcMarginProduct(fc_dim, num_classes, s=s, m=margin)
    elif loss_module == 'arcface2':
      self.face_margin_product = ArcMarginProduct2(fc_dim, num_classes, s=s, m=margin)
    else:
      raise ValueError(loss_module)

  def extract_feature(self, x):
    x = self.backbone.extract_features(x)
    x = self.pool(x)
    x = x.view(x.size(0), -1)
    x = self.fc(x)
    x = self.bn(x)
    return x

  def forward(self, x, label, **kargs):
    feature = self.extract_feature(x)
    out_face = self.face_margin_product(feature, label)
    return out_face, feature

def class_efficientnet_b5_gem_fc_arcface_1head(**kwargs):
  args = kwargs['args']
  model = ClsClass(args, feature_net='efficientnet_b5', loss_module='arcface', s=30, margin=0.3)
  return model

def class_efficientnet_b5_gem_fc_arcface2_1head(**kwargs):
  args = kwargs['args']
  model = ClsClass(args, feature_net='efficientnet_b5', loss_module='arcface2', s=30, margin=0.3)
  return model

def class_efficientnet_b6_gem_fc_arcface2_1head(**kwargs):
  args = kwargs['args']
  model = ClsClass(args, feature_net='efficientnet_b6', loss_module='arcface2', s=30, margin=0.3)
  return model

def class_efficientnet_b7_gem_fc_arcface2_1head(**kwargs):
  args = kwargs['args']
  model = ClsClass(args, feature_net='efficientnet_b7', loss_module='arcface2', s=30, margin=0.3)
  return model
