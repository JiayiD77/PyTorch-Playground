import torch
import torchvision
from torchvision import datasets
from torchvision import transforms
from torch.utils.data import DataLoader

BATCH_SIZE = 64


data_path = '..\Datasets\FOOD101'

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

train_data = datasets.Food101(root=data_path, transform=transform, download=True)
test_data = datasets.Food101(root=data_path, split='test', transform=transform, download=True)

train_data_loader = DataLoader(train_data, BATCH_SIZE, shuffle=True)
test_data_loader = DataLoader(test_data, BATCH_SIZE, shuffle=True)

# print(f"Total train batch: {len(train_data_loader)}")
# print(f"Total test batch: {len(test_data_loader)}")
# print(train_data.classes)

