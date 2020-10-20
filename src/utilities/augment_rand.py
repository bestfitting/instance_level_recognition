import cv2
import numpy as np
from PIL import Image, ImageOps, ImageEnhance, ImageDraw

fast_randaugment_list = [
  ('Identity', 0, 1),
  ('AutoContrast', 0, 10),
  ('Block_fade', 0, 0.5),
  ('Brightness', 0.5, 1.5),
  ('Color', 0.0, 2.0),
  ('Contrast', 0.0, 2.0),
  ('Cutout', 0, 0.5),
  ('Rotate', -20, 20),
  ('ShearX', 0., 0.1),
  ('ShearY', 0., 0.1),
  ('TranslateX', 0., 0.25),
  ('TranslateY', 0., 0.25),
]

def apply_op(image, op, severity):
  pil_img = Image.fromarray(image)
  pil_img = eval(op)(pil_img, severity)
  return np.asarray(pil_img)

def randaugment_base(augment_list, img, n, m, div=10, prob=1.):
  # ops = np.random.choice(augment_list, size=n)
  ops_idx = np.random.choice(len(augment_list), replace=False, size=n)
  ops = np.array(augment_list)[ops_idx]
  for op, minseverity, maxseverity in ops:
    minseverity = float(minseverity)
    maxseverity = float(maxseverity)
    severity = (float(m) / div) * float(maxseverity - minseverity) + minseverity
    if np.random.random() < prob:
      img = apply_op(img, str(op), severity)
  return img

def CutoutAbs(img, v):
  if v < 0:
    return img
  w, h = img.size
  x0 = np.random.uniform(w)
  y0 = np.random.uniform(h)

  x0 = int(max(0, x0 - v / 2.))
  y0 = int(max(0, y0 - v / 2.))
  x1 = min(w, x0 + v)
  y1 = min(h, y0 + v)

  xy = (x0, y0, x1, y1)
  color = (125, 123, 114)
  img = img.copy()
  ImageDraw.Draw(img).rectangle(xy, color)
  return img

def do_random_block_fade(image, magnitude=0.5):
  size = [0.1, magnitude]
  height, width = image.shape[:2]

  # get bounding box
  m = image.copy()
  cv2.rectangle(m, (0, 0), (height, width), 1, 5)
  m = image < 0.5
  if m.sum() == 0: return image

  m = np.where(m)
  y0, y1, x0, x1 = np.min(m[0]), np.max(m[0]), np.min(m[1]), np.max(m[1])
  w = x1 - x0
  h = y1 - y0
  if w * h < 10: return image

  ew, eh = np.random.uniform(*size, 2)
  ew = int(ew * w)
  eh = int(eh * h)

  ex = np.random.randint(0, w - ew) + x0
  ey = np.random.randint(0, h - eh) + y0

  image[ey:ey + eh, ex:ex + ew] *= np.random.uniform(0.1, 0.5)  # 1 #
  image = np.clip(image, 0, 1)
  return image

def Identity(img, _):
  return img

def AutoContrast(img, v):
  return ImageOps.autocontrast(img, v)

def Rotate(img, v):
  if np.random.random() > 0.5:
    v = -v
  return img.rotate(v)

def Color(img, v):
  return ImageEnhance.Color(img).enhance(v)

def Contrast(img, v):
  return ImageEnhance.Contrast(img).enhance(v)

def Brightness(img, v):
  return ImageEnhance.Brightness(img).enhance(v)

def ShearX(img, v):
  if np.random.random() > 0.5:
    v = -v
  return img.transform(img.size, Image.AFFINE, (1, v, 0, 0, 1, 0))

def ShearY(img, v):
  if np.random.random() > 0.5:
    v = -v
  return img.transform(img.size, Image.AFFINE, (1, 0, 0, v, 1, 0))

def TranslateX(img, v):
  if np.random.random() > 0.5:
    v = -v
  v = v * img.size[0]
  return img.transform(img.size, Image.AFFINE, (1, 0, v, 0, 1, 0))

def TranslateY(img, v):
  if np.random.random() > 0.5:
    v = -v
  v = v * img.size[1]
  return img.transform(img.size, Image.AFFINE, (1, 0, 0, 0, 1, v))

def Cutout(img, v):
  if v <= 0.:
    return img

  v = v * img.size[0]
  return CutoutAbs(img, v)

def Block_fade(img, v):
  img = np.asarray(img)
  img = img / 255.
  img = do_random_block_fade(img, v)
  img = img * 255.
  return img.astype(np.uint8)
