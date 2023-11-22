import os
import torch
from torch import nn
from torchvision import transforms
import data_setup, engine, models, utils


def train():
    torch.manual_seed(42) 
    torch.cuda.manual_seed(42)

    # Hyperparameters
    NUM_EPOCHS = 5
    BATCH_SIZE = 32
    HIDDEN_UNITS = 10
    LEARNING_RATE = 0.001

    # paths
    train_dir = "D:/PyTorch-Playground/Datasets/pizza_steak_sushi/train"
    test_dir = "D:/PyTorch-Playground/Datasets/pizza_steak_sushi/test"

    # CUDA setup
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Transform
    data_transform = transforms.Compose([ 
    transforms.Resize((64, 64)),
    transforms.ToTensor(),
    ])

    # Create dataloaders
    train_dataloader, test_dataloader, class_names = data_setup.create_dataloaders(train_dir,
                                                                            test_dir,
                                                                            transform=data_transform,
                                                                            batch_size=BATCH_SIZE)

    # Recreate an instance of TinyVGG
    model = models.TinyVGG(input_shape=3,
                    hidden_units=HIDDEN_UNITS,
                    output_shape=len(class_names)).to(device)


    # Setup loss function and optimizer
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(params=model.parameters(), lr=LEARNING_RATE)

    # Start the timer
    from timeit import default_timer as timer 
    start_time = timer()

    # Train model_0 
    engine.train(model,
            train_dataloader,
            test_dataloader,
            optimizer,
            loss_fn,
            epochs=NUM_EPOCHS,
            device=device)

    # End the timer and print out how long it took
    end_time = timer()
    print(f"[INFO] Total training time: {end_time-start_time:.3f} seconds")

if __name__ == '__main__':
    train()