import os
import torch
from torch import nn
import torchvision
from torchinfo import summary


device = 'cuda' if torch.cuda.is_available() else 'cpu'

def set_seeds(seed: int=42):
    """Sets the random seed

    Args:
        seed (int, optional): random seed to set. Defaults to 42.
    """
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

def check_image_data_property(image_data_point):
    class_names = CIFAR10_data_train.classes
    print(f'Single image shape: {image_data_point[0].shape}')
    print(f'Image Class: {class_names[image_data_point[1]]}')

# Import model
weights = torchvision.models.EfficientNet_B2_Weights.DEFAULT
model = torchvision.models.efficientnet_b2(weights=weights).to(device)

# Download CIFAR10 Datasets
path = '.\Datasets\CIFAR10'
transforms = weights.transforms()
CIFAR10_data_train = torchvision.datasets.CIFAR10(root=path, transform=transforms, download=True)
CIFAR10_data_test = torchvision.datasets.CIFAR10(root=path, transform=transforms, train=False, download=True)
print(f'Number of training samples: {len(CIFAR10_data_train)}')
print(f'Number of testing samples: {len(CIFAR10_data_test)}')
print(f'\n{"-" * 80}\n')

# Check the data
check_image_data_property(CIFAR10_data_train[0])
print(f'\n{"-" * 80}\n')

# Create dataloader
train_data_loader = torch.utils.data.DataLoader(CIFAR10_data_train, 
                                                batch_size = 64,
                                                shuffle=True,
                                                num_workers=os.cpu_count())
test_data_loader = torch.utils.data.DataLoader(CIFAR10_data_test, 
                                               batch_size = 64,
                                               shuffle=False,
                                               num_workers=os.cpu_count())

print(f'Train batches: {len(train_data_loader)}')
print(f'Test batches: {len(test_data_loader)}')

# Modify the efficientnet model
for param in model.features.parameters():
    param.requires_grad = False

classifier = nn.Sequential(
    nn.Dropout(p=0.3, inplace=True),
    nn.Linear(in_features=1408, out_features=10, bias=True)
).to(device)

model.classifier = classifier

model_summary = summary(model,
                        input_size=(64, 3, 288, 288),
                        verbose=0,
                        )
print(model_summary)