import os
ope = os.path.exists
opj = os.path.join
import numpy as np
import socket
import warnings
warnings.filterwarnings('ignore')

sk = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
hostname = socket.gethostname()

RESULT_DIR     = '/data4/data/retrieval2020/result'
DATA_DIR       = '/data5/data/landmark2020'
PRETRAINED_DIR = '/data5/data/pretrained'

PI  = np.pi
INF = np.inf
EPS = 1e-12
NUM_CLASSES = 81313

ID      = 'id'
TARGET  = 'landmark_id'
CLUSTER = 'cluster'
SCALE   = 'scale'
CTARGET = 'landmarks'
