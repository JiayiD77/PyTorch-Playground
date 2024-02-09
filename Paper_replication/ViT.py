import torch
import torchvision
from torch import nn
from torchvision import datasets
from torchvision import transforms
from torch.utils.data import DataLoader, SubsetRandomSampler

BATCH_SIZE = 64
PATCH_SIZE = 16

data_path = '..\Datasets\FOOD101'

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

train_data = datasets.Food101(root=data_path, transform=transform, download=True)
test_data = datasets.Food101(root=data_path, split='test', transform=transform, download=True)

# Use sampler to train and test on a tenth of the data
train_sampler = SubsetRandomSampler(indices=[i for i in range(len(train_data)) if (i % 10) == 0])
test_sampler = SubsetRandomSampler(indices=[i for i in range(len(test_data)) if (i % 10) == 0])

train_data_loader = DataLoader(train_data, BATCH_SIZE, sampler=train_sampler)
test_data_loader = DataLoader(test_data, BATCH_SIZE, sampler=test_sampler)

# Uncomment to train on the whole dataset
# train_data_loader = DataLoader(train_data, BATCH_SIZE, shuffle=True)
# test_data_loader = DataLoader(test_data, BATCH_SIZE, shuffle=True)

# print(f"Total train batch: {len(train_data_loader)}")
# print(f"Total test batch: {len(test_data_loader)}")
# print(f"Total classes: {len(train_data.classes)}")

embeddings_size = PATCH_SIZE**2 * 3
conv2d_layer = nn.Conv2d(in_channels=3, 
                         out_channels=embeddings_size,
                         kernel_size=PATCH_SIZE,
                         stride=PATCH_SIZE,
                         padding=0)

train_batch = next(iter(train_data_loader))
# print(len(train_batch))
# print(train_batch[0].shape)
# print(train_batch[1].shape)

single_image = train_batch[0][0].unsqueeze(dim=0)
embedded_single_image = conv2d_layer(single_image)
print(embedded_single_image.shape)