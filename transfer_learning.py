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