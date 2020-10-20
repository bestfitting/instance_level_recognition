import sys
sys.path.insert(0, '..')

from layers.scheduler import *
from layers.loss import *
from layers.backbone.resnet import *
from config.config import *
from layers.pooling import gem
from layers.metric_learning import *
from utilities.model_util import load_pretrained

model_names = {
  'resnet18': 'resnet18-5c106cde.pth',
  'resnet34': 'resnet34-333f7ec4.pth',
  'resnet50': 'resnet50-19c8e357.pth',
  'resnet101': 'resnet101-5d3b4d8f.pth',
  'resnet152': 'resnet152-b121ed2d.pth',
}
## net  ######################################################################
class ResnetClass(nn.Module):

  def __init__(self,
               args,
               feature_net='resnet101',
               loss_module='arcface',
               s=30.0,
               margin=0.3,
               ):
    super().__init__()
    num_classes = args.num_classes

    if feature_net == 'resnet18':
      self.backbone = resnet18()
      self.EX = 1
    elif feature_net == 'resnet34':
      self.backbone = resnet34()
      self.EX = 1
    elif feature_net == 'resnet50':
      self.backbone = resnet50()
      self.EX = 4
    elif feature_net == 'resnet101':
      self.backbone = resnet101()
      self.EX = 4
    elif feature_net == 'resnet152':
      self.backbone = resnet152()
      self.EX = 4

    self.backbone = load_pretrained(self.backbone,
                                    f'{PRETRAINED_DIR}/{model_names[feature_net]}',
                                    strict=True, can_print=args.can_print)
    self.in_channels = args.in_channels

    self.pool = gem
    fc_dim = 512
    self.fc = nn.Linear(512 * self.EX, fc_dim)
    self.bn = nn.BatchNorm1d(fc_dim)

    if loss_module == 'arcface':
      self.face_margin_product = ArcMarginProduct(fc_dim, num_classes, s=s, m=margin)
    elif loss_module == 'arcface2':
      self.face_margin_product = ArcMarginProduct2(fc_dim, num_classes, s=s, m=margin)
    else:
      raise ValueError(loss_module)

  def extract_feature(self, x):
    x = self.backbone.conv1(x)
    x = self.backbone.bn1(x)
    x = self.backbone.relu(x)
    x = self.backbone.maxpool(x)
    e2 = self.backbone.layer1(x)
    e3 = self.backbone.layer2(e2)
    e4 = self.backbone.layer3(e3)
    e5 = self.backbone.layer4(e4)
    x = self.pool(e5)
    x = x.view(x.size(0), -1)
    x = self.fc(x)
    x = self.bn(x)
    return x

  def forward(self, x, label, **kargs):
    feature = self.extract_feature(x)
    out_face = self.face_margin_product(feature, label)
    return out_face, feature

def class_resnet152_gem_fc_arcface_1head(**kwargs):
  args = kwargs['args']
  model = ResnetClass(args, feature_net='resnet152', loss_module='arcface', s=30.0, margin=0.3)
  return model
