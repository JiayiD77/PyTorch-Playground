import os
import torch
import torchvision
from torch import nn
from torchinfo import summary
from torch.utils.tensorboard import SummaryWriter
from tqdm.auto import tqdm
from datetime import datetime

device = 'cuda' if torch.cuda.is_available() else 'cpu'

def set_seeds(seed: int=42):
    """Sets the random seed

    Args:
        seed (int, optional): random seed to set. Defaults to 42.
    """
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

def check_image_data_property(image_data_point, class_names):
    print(f'Single image shape: {image_data_point[0].shape}')
    print(f'Image Class: {class_names[image_data_point[1]]}')

def create_writer(experiment_name, model_name, extra=None):
    time_stamp = datetime.now().strftime("%Y-%m-%d-%H-%M")
    if extra:
         log_dir = os.path.join('runs', time_stamp, experiment_name, model_name, extra)
    else:
        log_dir = os.path.join('runs', time_stamp, experiment_name, model_name)
    writer = SummaryWriter(log_dir=log_dir)
    return writer

def main():
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
    class_names = CIFAR10_data_train.classes
    check_image_data_property(CIFAR10_data_train[0], class_names)
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

    # Loss and optimizer
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    # Train
    set_seeds()

    Epochs = 5
    writer = create_writer('experiment_tracking', 'EfficientNet_B2', '10_epoch')
    
    for epoch in tqdm(range(Epochs)):
        model.train()
        train_loss, train_acc = 0, 0
        for batch, (X, y) in enumerate(train_data_loader):
            X, y = X.to(device), y.to(device)
            y_pred = model(X)
            loss = loss_fn(y_pred, y)
            train_loss += loss.item()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            y_pred_class = torch.argmax(torch.softmax(y_pred, dim=1), dim=1)
            train_acc += (y_pred_class == y).sum().item() / len(y)
        train_loss = train_loss / len(train_data_loader)
        train_acc = train_acc / len(train_data_loader)
        print(f'Epoch {epoch}:\n')
        print(f'Train loss: {train_loss:.4f} | Train acc: {train_acc:.4f}\n')
        
        model.eval()
        test_loss, test_acc = 0, 0
        with torch.inference_mode():
            for batch, (X, y) in enumerate(test_data_loader):
                X, y = X.to(device), y.to(device)
                y_pred = model(X)
                loss = loss_fn(y_pred, y)
                test_loss += loss.item()
                y_pred_class = torch.argmax(torch.softmax(y_pred, dim=1), dim=1)
                test_acc += (y_pred_class == y).sum().item() / len(y)
            test_loss = test_loss / len(test_data_loader)
            test_acc = test_acc / len(test_data_loader)
            print(f'Test loss: {test_loss:.4f} | Test acc: {test_acc:.4f}\n')

            writer.add_scalars(main_tag='Loss', tag_scalar_dict={'train_loss':train_loss, 'test_loss':test_loss}, global_step=epoch)
            writer.add_scalars(main_tag='Accuracy', tag_scalar_dict={'train_acc':train_acc, 'test_acc':test_acc}, global_step=epoch)
        
    writer.close()

if __name__=="__main__":
    main()