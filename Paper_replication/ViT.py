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

print(f'Shape of image after patch embedding: {embedded_single_image.shape}') 

class MultiHeadSelfAttentionBlock(nn.Module):
    def __init__(self,
                 embedding_dim,
                 num_heads=12,
                 attn_dropout=0):
        super().__init__()   
        
        self.layer_norm = nn.LayerNorm(normalized_shape=embedding_dim)
        self.multihead_attn = nn.MultiheadAttention(embed_dim=embedding_dim,
                                                    num_heads=num_heads,
                                                    dropout=attn_dropout,
                                                    batch_first=True)
        
    def forward(self, x):
        x = self.layer_norm(x)
        attn_ouput, _ = self.multihead_attn(query=x,
                                            key=x,
                                            value=x,
                                            need_weights=False)
        return attn_ouput

attn_block = MultiHeadSelfAttentionBlock(embedding_dim=EMBEDDING_SIZE)
img_attn = attn_block(embedded_single_image)

print(f'Shape of image after passing through attention layer: {img_attn.shape}')   

class MLP(nn.Module):
    def __init__(self,
                 embedding_dim,
                 mlp_size,
                 dropout=0.1):
        super().__init__()
        
        self.layer_norm = nn.LayerNorm(normalized_shape=embedding_dim)
        self.mlp = nn.Sequential(
            nn.Linear(in_features=embedding_dim, out_features=mlp_size),
            nn.GELU(),
            nn.Dropout(p=dropout),
            nn.Linear(in_features=mlp_size, out_features=embedding_dim),
            nn.Dropout(p=dropout)
        )
    
    def forward(self, x):
        x = self.layer_norm(x)
        x = self.mlp(x)
        return x

mlp = MLP(embedding_dim=EMBEDDING_SIZE, mlp_size=3072)    
img_mlp = mlp(img_attn)

print(f'Shape of image after passing through mlp: {img_mlp.shape}')  

class TransformerEncoderBlock(nn.Module):
    def __init__(self,
                 embedding_dim,
                 num_heads,
                 attn_dropout,
                 mlp_size,
                 dropout):
        super().__init__()
        
        self.msa_block = MultiHeadSelfAttentionBlock(embedding_dim=embedding_dim, 
                                                     num_heads=num_heads, 
                                                     attn_dropout=attn_dropout)
        self.mlp_block = MLP(embedding_dim=embedding_dim, 
                             mlp_size=mlp_size,
                             dropout=dropout)
    
    def forward(self, x):
        x = self.msa_block(x) + x
        x = self.mlp_block(x) + x
        return x
    
class ViT(nn.Module):
    def __init__(self,
                 in_channels:int=3,
                 patch_size:int=16,
                 num_transformer_layers:int=12,
                 embedding_dim:int=768,
                 num_heads:int=12,
                 mlp_size:int=3072,
                 attn_dropout:int=0,
                 mlp_dropout:int=0.1,
                 embedding_dropout:int=0.1,
                 num_classes:int=1000):
        super().__init__()
        
        self.patch_embedding = PatchEmbedding(in_channels=in_channels,
                                                patch_size=patch_size,
                                                embedding_dim=embedding_dim)
        
        self.embedding_dropout = nn.Dropout(embedding_dropout)
        
        self.encoder_block = nn.Sequential(
            *[TransformerEncoderBlock(embedding_dim=embedding_dim,
                                      num_heads=num_heads,
                                      attn_dropout=attn_dropout,
                                      mlp_size=mlp_size,
                                      dropout=mlp_dropout) for _ in range(num_transformer_layers)]
        )
        
        self.classifier = nn.Sequential(
            nn.LayerNorm(normalized_shape=embedding_dim),
            nn.Linear(in_features=embedding_dim,
                      out_features=num_classes)
        )
        
    def forward(self, x):
        x = self.patch_embedding(x)
        x = self.encoder_block(x)
        x = self.classifier(x)
        return x

model = ViT(num_classes=10)
print(f"Input image shape: {single_image.shape}")
output = model(single_image.unsqueeze(dim=0))
print(f"Output shape: {output.shape}")