import sys
sys.path.insert(0, '..')
from imgaug import augmenters as iaa

from config.config import *
from utilities.augment_rand import *
import random
import torchvision

def train_multi_augment1(image, **kwargs):
  seq = iaa.SomeOf(1, [
    iaa.Noop(),
    iaa.Fliplr(p=1),
  ])
  image = seq.augment_images([image])[0]
  return image

def train_multi_augment3(image, img_size=(600, 600), **kwargs):
  if np.random.random() < 0.5:
    image = random_crop_long_edge(image, img_size)
  else:
    image = cv2.resize(image, img_size)
  return image

def train_multi_augment3b(image, img_size=(600, 600), **kwargs):
  image = do_rand_aug(image, fast_randaugment_list)
  image = train_multi_augment3(image, img_size=img_size)
  return image

def do_rand_aug(image, randaugment_list):
  if np.random.random() < 0.03:
    if np.random.random() < 0.5:
      image = np.rot90(image, k=1)
    else:
      image = np.rot90(image, k=3)

  n = 3
  m = random.randint(1, 9)
  if image.max() <= 1:
    image = (image * 255).astype('uint8')
  image = randaugment_base(randaugment_list, image, n, m, div=10, prob=1)

  seq = iaa.OneOf([
    iaa.Noop(),
    iaa.Fliplr(p=1),
    iaa.Affine(scale=(0.75, 1.25)),
  ])
  image = seq.augment_images([image])[0]
  return image

def random_crop_long_edge(img, img_size=(600,600)):
  if img.max() <= 1:
    img = (img * 255).astype('uint8')
  size = (min(img.shape[:2]), min(img.shape[:2]))
  i = (0 if size[0] == img.shape[0]
       else np.random.randint(low=0,high=img.shape[0] - size[0]))
  j = (0 if size[1] == img.shape[1]
       else np.random.randint(low=0,high=img.shape[1] - size[1]))
  img = Image.fromarray(img)
  img = torchvision.transforms.functional.crop(img, i, j, size[1], size[0])
  img = np.asarray(img)
  img = cv2.resize(img, img_size)
  return img
