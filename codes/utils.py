import numpy as np
import random
import os

# Function to set the random seed for reproducibility
def set_seed(seed=42):
    """
    Set the random seed for reproducibility
    """
    np.random.seed(seed)
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)