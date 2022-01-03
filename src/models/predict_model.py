%load_ext autoreload
%autoreload 2
import tensorflow as tf
from pathlib import Path
import numpy as np
import sys
sys.path.insert(0, "..")

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1' 

from src.data import make_dataset