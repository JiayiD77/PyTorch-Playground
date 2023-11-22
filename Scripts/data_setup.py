import os

from torchvision import datasets, transforms
from torch.utils.data import DataLoader

NUM_WORKERS = os.cpu_count()

def create_dataloaders(
    train_dir: str,
    test_dir: str,
    transform: transforms.Compose,
    batch_size: int,
    num_workers: int=NUM_WORKERS
):
    """Create training and testing DataLoaders

    Args:
        train_dir (str): Path to training directory.
        test_dir (str): Path to testing directory.
        transform (transforms.Compose): Torchvision transforms to perform on data
        batch_size (int): Number of samples per batch
        num_workers (int, optional): Number of worker on dataloaders. Defaults to NUM_WORKERS.
    """
    # Create datasets
    train_data = datasets.ImageFolder(train_dir, transform=transform)
    test_data = datasets.ImageFolder(test_dir, transform=transform)
    
    # Get class names
    class_names = train_data.classes
    
    # Turn into dataloaders
    train_dataloader = DataLoader(
        train_data,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )
    
    test_dataloader = DataLoader(
        test_data,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    return train_dataloader, test_dataloader, class_names
    