from Utils import data_setup
import torchvision

weights = torchvision.models.EfficientNet_B0_Weights.DEFAULT
transform = weights.transforms()

print(transform)

train_dir = "Datasets/pizza_steak_sushi/train"
test_dir = "Datasets/pizza_steak_sushi/test"

train_dataloader, test_dataloader, class_names = data_setup.create_dataloaders(train_dir = train_dir,
                                                                               test_dir=test_dir,
                                                                               transform=transform,
                                                                               batch_size=32)

# Getting the pre-trained model from torchvision
model = torchvision.models.efficientnet_b0(weights=weights)

print("----------------------------------------------------------------")
print("Model Architecture")
print(model)
print("----------------------------------------------------------------")
print("Model Classifier")
print(model.classifier)

