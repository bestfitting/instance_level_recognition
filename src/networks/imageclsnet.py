from argparse import Namespace
from config.config import *
from networks.efficientnet_gem_fc_face import (
  class_efficientnet_b5_gem_fc_arcface_1head,
  class_efficientnet_b5_gem_fc_arcface2_1head,
  class_efficientnet_b6_gem_fc_arcface2_1head,
  class_efficientnet_b7_gem_fc_arcface2_1head,
)
from networks.resnet_gem_fc_face import class_resnet152_gem_fc_arcface_1head

def init_network(params):
  architecture = params.get('architecture', 'class_efficientnet_b7_gem_fc_arcface2_1head')
  args = Namespace(**{
    'num_classes': params.get('num_classes', 81313),
    'in_channels': params.get('in_channels', 3),
    'can_print': params.get('can_print', False),
  })
  net = eval(architecture)(args=args)
  return net
