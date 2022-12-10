import itertools

import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from pyselect.datasets import make_jordan_se2
from pyselect.model import RFFNet, RFFLayer
