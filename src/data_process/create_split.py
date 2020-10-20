import sys
sys.path.insert(0, '..')
import pandas as pd
from tqdm import tqdm
from config.config import *

def create_whole_train_split(train_meta, split_name):
  train_meta = train_meta.copy()
  split_dir = f'{DATA_DIR}/split/{split_name}'
  os.makedirs(split_dir, exist_ok=True)

  print('train nums: %s' % train_meta.shape[0])
  print('train label nums: %s' % train_meta[TARGET].nunique())
  train_meta['count'] = train_meta.groupby([TARGET])[ID].transform('count')
  litter_image_df = train_meta[train_meta['count'] < 200]
  train_rest_meta = train_meta[~train_meta[ID].isin(litter_image_df[ID].values)].reset_index(drop=True)

  idx = 0
  valid_indices = np.random.choice(len(train_rest_meta), 200, replace=False)
  valid_split_df = train_rest_meta.loc[valid_indices]
  train_indices = ~train_meta[ID].isin(valid_split_df[ID].values)
  train_split_df = train_rest_meta[train_indices]
  train_split_df = pd.concat((train_split_df, litter_image_df), ignore_index=True)

  fname = f'{split_dir}/random_train_cv{idx}.csv'
  print("train: create split file: %s; "% (fname))
  print(('nums: %d; label nums: %d; max label: %s')%
      (train_split_df.shape[0],train_split_df[TARGET].nunique(),train_split_df[TARGET].max()))
  train_split_df.to_csv(fname, index=False)
  print(train_split_df.head())

  fname = f'{split_dir}/random_valid_cv{idx}.csv'
  print("valid: create split file: %s; "% (fname))
  print(('nums: %d; label nums: %d; max label: %s') %
        (valid_split_df.shape[0],valid_split_df[TARGET].nunique(),valid_split_df[TARGET].max()))
  valid_split_df.to_csv(fname, index=False)
  print(valid_split_df.head())

def create_v2x_split():
  train_clean_df = pd.read_csv(f'{DATA_DIR}/raw/train_clean.csv', usecols=[TARGET])
  train_df = pd.read_csv(f'{DATA_DIR}/raw/train.csv', usecols=[ID, TARGET])
  train_df = train_df[train_df[TARGET].isin(train_clean_df[TARGET].unique())]

  landmark_mapping = {l: i for i, l in enumerate(np.sort(train_df[TARGET].unique()))}
  train_df[TARGET] = train_df[TARGET].map(landmark_mapping)

  idx = 0
  train_split_df = pd.read_csv(f'{DATA_DIR}/split/v2c/random_train_cv{idx}.csv')
  valid_split_df = pd.read_csv(f'{DATA_DIR}/split/v2c/random_valid_cv{idx}.csv')
  _train_df = train_df.set_index(ID)
  assert np.array_equal(_train_df.loc[train_split_df[ID].values, TARGET], train_split_df[TARGET])
  assert np.array_equal(_train_df.loc[valid_split_df[ID].values, TARGET], valid_split_df[TARGET])
  del _train_df

  train_df = train_df[~train_df[ID].isin(valid_split_df[ID])]
  train_split_df = pd.merge(train_df, train_split_df, on=[ID, TARGET], how='left')

  split_dir = f'{DATA_DIR}/split/v2x'
  os.makedirs(split_dir, exist_ok=True)

  fname = f'{split_dir}/random_train_cv{idx}.csv'
  print("train: create split file: %s; "% (fname))
  print(('nums: %d; label nums: %d') % (train_split_df.shape[0], train_split_df[TARGET].nunique()))
  train_split_df.to_csv(fname, index=False)
  print(train_split_df.head())

  fname = f'{split_dir}/random_valid_cv{idx}.csv'
  print("valid: create split file: %s; "% (fname))
  print(('nums: %d; label nums: %d') % (valid_split_df.shape[0], valid_split_df[TARGET].nunique()))
  valid_split_df.to_csv(fname, index=False)
  print(valid_split_df.head())

if __name__ == "__main__":
  print('%s: calling main function ... ' % os.path.basename(__file__))
  train_clean_df = pd.read_csv(f'{DATA_DIR}/raw/train_clean.csv')
  train_clean_df['count'] = [len(row.split(' ')) for row in train_clean_df['images'].values]
  train_clean_df[CTARGET] = train_clean_df[TARGET]
  train_clean_df[TARGET] = range(len(train_clean_df))
  images = []
  for _, row in tqdm(train_clean_df.iterrows(), total=len(train_clean_df)):
    label = row[TARGET]
    old_label = row[CTARGET]
    for file_id in row['images'].split(' '):
      images.append((file_id, label, old_label))

  dataset_df = pd.DataFrame(data=images, columns=[ID, TARGET, CTARGET])
  dataset_df = dataset_df.sample(len(dataset_df), replace=False, random_state=100).reset_index(drop=True)
  dataset_df.to_csv(f'{DATA_DIR}/split/train2020.csv', index=False)

  create_whole_train_split(dataset_df, split_name='v2c')
  create_v2x_split()
