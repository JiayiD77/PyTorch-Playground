from Utils import data_setup, engine

import torch
from torch import nn
import torchvision
from torchinfo import summary

from timeit import default_timer as timer

def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

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
    model = torchvision.models.efficientnet_b0(weights=weights).to(device)

    # print("----------------------------------------------------------------")
    # print("Model Architecture")
    # print(model)
    # print("----------------------------------------------------------------")
    # print("Model Classifier")
    # print(model.classifier)

    summary(model=model,
            input_size=(1, 3, 224, 224),
            col_names=['input_size', 'output_size', 'num_params', 'trainable'],
            col_width=15,
            row_settings=['var_names'])

    # Set the trained params to untrainable
    for params in model.features.parameters():
        params.requires_grad = False

    # Change the classifier
    torch.manual_seed(42)
    torch.cuda.manual_seed(42)

    model.classifier = nn.Sequential(
        nn.Dropout(0.2, inplace=True),
        nn.Linear(in_features=1280, out_features=len(class_names))
    ).to(device)

    print(model.classifier)
    print(f"Training on {device}.....")
    
    # Train the model
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    start_time = timer()
    results = engine.train(model=model,
                        train_dataloader=train_dataloader,
                        test_dataloader=test_dataloader,
                        optimizer=optimizer,
                        loss_fn=loss_fn,
                        epochs=10,
                        device=device)
    end_time = timer()
    print(f"Total training time: {end_time-start_time:.3f} seconds")

if __name__=="__main__":
    main()