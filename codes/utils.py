import numpy as np
import torch
import random
import os

# Function to set the random seed for reproducibility
def set_seed(seed=42):
    """
    Set the random seed for reproducibility
    """
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)