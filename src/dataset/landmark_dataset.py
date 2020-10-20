import sys
sys.path.insert(0, '..')
import torch
from torch.utils.data import Dataset
import pandas as pd
from albumentations import Normalize

from utilities.augment_util import *

class RetrievalDataset(Dataset):
  def __init__(self, args, split_file, transform, data_type='train'):
    self.args = args
    self.img_size = (args.img_size, args.img_size)
    self.transform = transform
    self.is_train = data_type == 'train'

    df = pd.read_csv(split_file)
    self.df = df
    if data_type == 'valid':
        self.df = self.df[:200]

    img_dir = f'{DATA_DIR}/images/train'
    self.do_print('img_dir %s' % img_dir)
    self.img_dir = img_dir

    if self.is_train:
      self.df = self.df.sample(len(self.df), replace=False).reset_index(drop=True)
      dataset_df = self.df
    else:
      dataset_df = self.df

    self.dataset_df = dataset_df
    self.do_resample()

  def do_resample(self):
    dataset_df = self.dataset_df
    self.x = dataset_df[ID].values
    self.y = dataset_df[TARGET].values

  def do_print(self, content):
    if self.args.can_print:
      print(content)

  def __len__(self):
    return len(self.x)

  def get_batch_images(self, idx, img_id, label):
    x = [img_id]
    y = [label]
    return x, y

  def __getitem__(self, idx):
    img_id = self.x[idx]
    label = self.y[idx]

    x, y = self.get_batch_images(idx, img_id, label)
    images = []
    for file_name in x:
      img_dir = self.img_dir
      boxes = None
      img_fname = f'{img_dir}/{file_name}.jpg'
      if not os.path.exists(img_fname):
        img_fname = f'{DATA_DIR}/images/test/{file_name}.jpg'
      image = cv2.imread(img_fname)
      image = image[..., ::-1]
      if self.transform is not None:
        image = self.transform(image, img_size=self.img_size, boxes=boxes)
      if image.shape[:2] != self.img_size:
        image = cv2.resize(image, self.img_size)

      if self.args.preprocessing == 1:
        norm = Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225), max_pixel_value=255.0)
        image = norm(image=image)['image']
      else:
        image = image / 255.0
      image = np.transpose(image, (2, 0, 1))
      image = torch.from_numpy(image).float()
      images.append(image)
    return images, y

  def on_epoch_end(self):
    if self.is_train:
      self.do_resample()
      idxes = np.random.choice(len(self.y), len(self.y), replace=False)
      self.x = np.array(self.x)[idxes]
      self.y = np.array(self.y)[idxes]

def image_collate(batch):
  batch_size = len(batch)
  images = []
  labels = []
  for b in range(batch_size):
    if batch[b][0] is None:
      continue
    else:
      images.extend(batch[b][0])
      labels.extend(batch[b][1])
  images = torch.stack(images, 0)
  labels = torch.from_numpy(np.array(labels))
  return images, labels
