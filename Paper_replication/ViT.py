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

train_batch = next(iter(train_data_loader))
# print(len(train_batch))
# print(train_batch[0].shape)
# print(train_batch[1].shape)

single_image = train_batch[0][0]
CHANNEL, HEIGHT, WIDTH = single_image.shape
EMBEDDING_SIZE = PATCH_SIZE**2 * 3

class PatchEmbedding(nn.Module):
    def __init__(self,
                 in_channels,
                 patch_size=PATCH_SIZE,
                 embedding_dim=EMBEDDING_SIZE):
        super().__init__()
        
        self.patch_size = patch_size
        self.embedding_dim = embedding_dim
        self.embedding_layer = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, 
                      out_channels=embedding_dim,
                      kernel_size=patch_size,
                      stride=patch_size,
                      padding=0),
            nn.Flatten(-2, -1)
        )
        
    def forward(self, x):
        embedded_image = self.embedding_layer(x)
        embedded_image = embedded_image.permute(0, 2, 1) # [B, P, D]
        
        # Add class embedding
        class_token = nn.Parameter(torch.rand(x.shape[0], 1, self.embedding_dim), requires_grad=True)
        embedded_image = torch.cat((class_token, embedded_image), dim=1)
        
        # Add position embedding
        patch_size_po = int(x.shape[2] * x.shape[3] / self.patch_size**2)
        position_embedding = nn.Parameter(torch.rand(1, patch_size_po+1, self.embedding_dim), requires_grad=True)
        embedded_image = embedded_image + position_embedding
        
        return embedded_image


patch_embedding = PatchEmbedding(in_channels=CHANNEL)
embedded_single_image = patch_embedding(single_image.unsqueeze(dim=0))
# class_token = nn.Parameter(torch.rand(embedded_single_image.shape[0], 1, EMBEDDING_SIZE), requires_grad=True)
# embedded_single_image = torch.cat((class_token, embedded_single_image), dim=1)
# patch_size = int(HEIGHT * WIDTH / PATCH_SIZE**2)
# position_embedding = nn.Parameter(torch.rand(1, patch_size+1, EMBEDDING_SIZE), requires_grad=True)
# embedded_single_image = embedded_single_image + position_embedding
print(embedded_single_image.shape)

