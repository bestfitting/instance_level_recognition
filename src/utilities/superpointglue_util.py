import cv2
import numpy as np
import torch

def process_resize(w, h, resize):
  assert (len(resize) > 0 and len(resize) <= 2)
  if len(resize) == 1 and resize[0] > -1:
    scale = resize[0] / max(h, w)
    w_new, h_new = int(round(w * scale)), int(round(h * scale))
  elif len(resize) == 1 and resize[0] == -1:
    w_new, h_new = w, h
  else:  # len(resize) == 2:
    w_new, h_new = resize[0], resize[1]

  # Issue warning if resolution is too small or too large.
  if max(w_new, h_new) < 160:
    print('Warning: input resolution is very small, results may vary')
  elif max(w_new, h_new) > 2000:
    print('Warning: input resolution is very large, results may vary')

  return w_new, h_new

def frame2tensor(frame, device='cuda'):
  return torch.from_numpy(frame / 255.).float()[None, None].to(device)

def read_image(path, resize, rotation, resize_float, device='cuda'):
  image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
  if image is None:
    return None, None, None
  w, h = image.shape[1], image.shape[0]
  w_new, h_new = process_resize(w, h, resize)
  scales = (float(w) / float(w_new), float(h) / float(h_new))

  if resize_float:
    image = cv2.resize(image.astype('float32'), (w_new, h_new))
  else:
    image = cv2.resize(image, (w_new, h_new)).astype('float32')

  if rotation != 0:
    image = np.rot90(image, k=rotation)
    if rotation % 2:
      scales = scales[::-1]

  inp = frame2tensor(image, device=device)
  return image, inp, scales
