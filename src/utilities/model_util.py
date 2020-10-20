import sys
sys.path.insert(0, '..')
import pandas as pd
import torch
from torch.nn import DataParallel
from apex.parallel import DistributedDataParallel

from config.config import *

def load_pretrained_state_dict(net, load_state_dict, strict=False, can_print=True, extend_W=None, num_classes=-1):
  if 'epoch' in load_state_dict and can_print:
    epoch = load_state_dict['epoch']
    print(f'load epoch:{epoch:.2f}')
  if 'state_dict' in load_state_dict:
    load_state_dict = load_state_dict['state_dict']
  elif 'model_state_dict' in load_state_dict:
    load_state_dict = load_state_dict['model_state_dict']
  elif 'model' in load_state_dict:
    load_state_dict = load_state_dict['model']
  if type(net) == DataParallel or type(net) == DistributedDataParallel:
    state_dict = net.module.state_dict()
  else:
    state_dict = net.state_dict()

  new_load_state_dict = dict()
  for key in load_state_dict.keys():
    if key.startswith('module.'):
      dst_key = key.replace('module.', '')
    else:
      dst_key = key
    new_load_state_dict[dst_key] = load_state_dict[key]
  load_state_dict = new_load_state_dict

  if extend_W is not None:
    extend_key = 'face_margin_product.W'
    if extend_key not in load_state_dict:
      raise Exception(f'{extend_key} is not in load_state_dict')
    if extend_W == 'nolandmark':
      old_landmarks = np.arange(load_state_dict[extend_key].size(0))
      new_landmarks = np.concatenate([old_landmarks, [np.max(old_landmarks) + 1]])
    else:
      arr = np.load(f'{DATA_DIR}/input/landmarks_mapping_{extend_W}.npz', allow_pickle=True)
      old_landmarks = arr['old_landmarks']
      new_landmarks = arr['new_landmarks']
    if load_state_dict[extend_key].size(0) != num_classes:
      print(f'{extend_key} shape: {len(old_landmarks)} -> {len(new_landmarks)}')
      load_state_dict = extend_model_weight(load_state_dict, old_landmarks, new_landmarks, key=extend_key)

  for key in list(load_state_dict.keys()):
    if key not in state_dict:
      if key == 'maxpool.1.filt':
        state_dict['maxpool.0.filt'] = load_state_dict[key]
        state_dict['maxpool.2.filt'] = load_state_dict[key]
        print('weight maxpool.1.filt --> maxpool.0.filt AND maxpool.2.filt')
      elif strict:
        raise Exception(f'not in {key}')
      if can_print:
        print('not in', key)
      continue
    if load_state_dict[key].size() != state_dict[key].size():
      if strict:
        raise Exception(f'size not the same {key}')
      if ('last_linear' in key or 'attention' in key) and (load_state_dict[key].size()[1:] == state_dict[key].size()[1:]):
        min_channel = min(state_dict[key].size(0), load_state_dict[key].size(0))
        state_dict[key][:min_channel] = load_state_dict[key][:min_channel]
      elif can_print:
        print('size not the same', key)
      continue
    state_dict[key] = load_state_dict[key]
  if type(net) == DataParallel or type(net) == DistributedDataParallel:
    net.module.load_state_dict(state_dict)
  else:
    net.load_state_dict(state_dict)
  return net

def extend_model_weight(state_dict, old_landmarks, new_landmarks, key='face_margin_product.W'):
  old_landmarks = np.sort(old_landmarks)
  new_landmarks = np.sort(new_landmarks)

  W = state_dict[key]
  assert W.size(0) == len(old_landmarks)
  assert len(W.size()) == 2
  new_W = torch.zeros((len(new_landmarks), W.size(1)), dtype=W.dtype)
  new_W[:, :] = W.mean(dim=0)

  old_ts = pd.Series(index=old_landmarks, data=np.arange(len(old_landmarks)))
  new_ts = pd.Series(index=new_landmarks, data=np.arange(len(new_landmarks)))
  intersection_landmarks = np.sort(list(set.intersection(set(old_landmarks), set(new_landmarks))))
  new_W[new_ts[intersection_landmarks].tolist(), :] = W[old_ts[intersection_landmarks].tolist(), :]

  state_dict[key] = new_W
  return state_dict

def load_pretrained(net, pretrained_file, strict=False,can_print=True):
  if can_print:
    print(f'load pretrained file: {pretrained_file}')
  load_state_dict = torch.load(pretrained_file)
  net = load_pretrained_state_dict(net, load_state_dict, strict=strict, can_print=can_print)
  return net
