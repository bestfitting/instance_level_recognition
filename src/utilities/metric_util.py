import sys
sys.path.insert(0, '..')
from pathlib import Path
from scipy.spatial import distance
import pandas as pd
import cv2
import torch
from layers.normalization import L2N
from config.config import *
from torch.autograd import Variable
from torch.utils.data import Dataset, DataLoader, SequentialSampler
from albumentations import Normalize

def MeanAveragePrecision(predictions, retrieval_solution, max_predictions=100, save_perimg_score=False):
  """Computes mean average precision for retrieval prediction.

  Args:
    predictions: Dict mapping test image ID to a list of strings corresponding
      to index image IDs.
    retrieval_solution: Dict mapping test image ID to list of ground-truth image
      IDs.
    max_predictions: Maximum number of predictions per query to take into
      account. For the Google Landmark Retrieval challenge, this should be set
      to 100.

  Returns:
    mean_ap: Mean average precision score (float).

  Raises:
    ValueError: If a test image in `predictions` is not included in
      `retrieval_solutions`.
  """
  # Compute number of test images.
  num_test_images = len(retrieval_solution.keys())

  # Loop over predictions for each query and compute mAP.
  mean_ap = 0.0
  score_map = {}
  for key, prediction in predictions.items():
    if key not in retrieval_solution:
      raise ValueError('Test image %s is not part of retrieval_solution' % key)

    # Loop over predicted images, keeping track of those which were already
    # used (duplicates are skipped).
    ap = 0.0
    already_predicted = set()
    num_expected_retrieved = min(len(retrieval_solution[key]), max_predictions)
    num_correct = 0
    for i in range(min(len(prediction), max_predictions)):
      if prediction[i] not in already_predicted:
        if prediction[i] in retrieval_solution[key]:
          num_correct += 1
          ap += num_correct / (i + 1)
        already_predicted.add(prediction[i])

    ap /= num_expected_retrieved
    mean_ap += ap
    score_map[key] = ap

  mean_ap /= num_test_images
  if save_perimg_score:
    return mean_ap, score_map
  else:
    return mean_ap

def generate_score_by_model(model, img_size=None, scale=None, selected_num=200, batch_size=1, preprocessing=False):
  # assert (img_size is None and scale is not None) or (img_size is not None and scale is None)
  K = 100
  QUERY_IMAGE_DIR = f'{DATA_DIR}/images/test'
  INDEX_IMAGE_DIR = f'{DATA_DIR}/images/index'
  solution_df = pd.read_csv(f'{DATA_DIR}/raw/retrieval_solution_v2.1.csv')
  private_df = solution_df[solution_df['Usage'] == 'Private']
  private_solution = {}
  private_test_img_ids = []
  private_index_img_ids = []

  for i, row in private_df.iterrows():
    private_solution[row['id']] = row['images'].split(' ')
    private_test_img_ids.append(row['id'])
    private_index_img_ids.extend(row['images'].split(' '))
    if len(private_test_img_ids) >= selected_num:
      break

  private_test_img_ids = list(set(private_test_img_ids))
  private_index_img_ids = list(set(private_index_img_ids))

  class _TestDataset(Dataset):
    def __init__(self, image_paths, img_size=None):
      self.image_paths = image_paths
      self.img_size = img_size

    def __len__(self):
      return len(self.image_paths)

    def __getitem__(self, idx):
      image_path = self.image_paths[idx]
      image = cv2.imread(str(image_path))
      image = image[..., ::-1]
      image = cv2.resize(image, self.img_size)
      if preprocessing == 1:
        norm = Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225), max_pixel_value=255.0)
        image = norm(image=image)['image']
      else:
        image = image / 255.0
      image = np.transpose(image, (2, 0, 1))
      image_tensor = torch.from_numpy(image).float()
      return image_tensor

  def _create_dataset(image_paths, img_size, batch_size):
    dataset = _TestDataset(image_paths, img_size)
    data_loader = DataLoader(
      dataset,
      sampler=SequentialSampler(dataset),
      batch_size=batch_size,
      drop_last=False,
      num_workers=4,
      pin_memory=True,
    )
    return data_loader

  def _get_embedding(image_tensor):
    image_tensor = Variable(image_tensor.cuda())
    embedding = model.module.extract_feature(image_tensor)
    embedding = L2N()(embedding)
    return embedding

  def _get_id(image_path: Path):
    return int(image_path.name.split('.')[0], 16)

  def _get_embeddings(image_root_dir: str):
    if image_root_dir.count('test') > 0:
      image_paths = [Path(f'{image_root_dir}/{img_id}.jpg') for img_id in private_test_img_ids]
    else:
      image_paths = [Path(f'{image_root_dir}/{img_id}.jpg') for img_id in private_index_img_ids]
    dataloader = _create_dataset(image_paths, img_size, batch_size)
    embeddings = []
    for image_tensor in dataloader:
      embedding = _get_embedding(image_tensor)
      embedding = embedding.cpu().detach().numpy()
      embeddings.extend(embedding)
    ids = [_get_id(image_path) for image_path in image_paths]
    return ids, embeddings

  def _to_hex(image_id: int) -> str:
    return '{0:0{1}x}'.format(image_id, 16)

  def _get_metrics(predictions, solution):
    relevant_predictions = {}

    for key in solution.keys():
      if key in predictions:
        relevant_predictions[key] = predictions[key]

    # Mean average precision.
    mean_average_precision = MeanAveragePrecision(relevant_predictions, solution, max_predictions=K)

    return mean_average_precision

  query_ids, query_embeddings = _get_embeddings(QUERY_IMAGE_DIR)
  index_ids, index_embeddings = _get_embeddings(INDEX_IMAGE_DIR)
  distances = distance.cdist(np.array(query_embeddings), np.array(index_embeddings), 'euclidean')
  predicted_positions = np.argpartition(distances, K, axis=1)[:, :K]

  predictions = {}
  for i, query_id in enumerate(query_ids):
    nearest = [(index_ids[j], distances[i, j]) for j in predicted_positions[i]]
    nearest.sort(key=lambda x: x[1])
    prediction = [_to_hex(index_id) for index_id, d in nearest]
    predictions[_to_hex(query_id)] = prediction
  score = _get_metrics(predictions, private_solution)
  return score
