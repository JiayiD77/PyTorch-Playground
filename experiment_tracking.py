import torch
import torchvision
from torch import nn

device = 'cuda' if torch.cuda.is_available() else 'cpu'

def set_seeds(seed: int=42):
    """Sets the random seed

    Args:
        seed (int, optional): random seed to set. Defaults to 42.
    """
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

