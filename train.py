from torch.utils.data import DataLoader
from torch import optim
from torch import nn
from torchvision.transforms import RandomRotation
from torch.optim.lr_scheduler import StepLR
from model import Resnet34
from mask_dataset import MASK
import torch
from torchvision import transforms
import matplotlib.pyplot as plt
import os


def train(data_dir, epochs):
    train_transforms = transforms.Compose(
        [
            transforms.Resize(size=100),
            transforms.RandomResizedCrop(50),
            RandomRotation(30),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
         ]
    )
    validation_transforms = transforms.Compose(
        [
            transforms.Resize(size=100),
            transforms.CenterCrop(size=50),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ]
    )

    model = Resnet34()
    train_loss = []
    valid_loss = []
    criterion = nn.CrossEntropyLoss()

    train_dataset = MASK(data_dir, "train", train_transforms)
    # optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9, weight_decay=0.001)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    scheduler = StepLR(optimizer, step_size=300, gamma=0.1)

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=128, shuffle=True, num_workers=4)
    validation_dataset = MASK(data_dir, "validation", validation_transforms)
    validation_loader = torch.utils.data.DataLoader(validation_dataset, batch_size=128, shuffle=True, num_workers=4)

    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    model = model.to(device)

    for e in range(epochs):
        print(f"Staring epoch {e}/{epochs}")
        cost_train = 0
        total = 0
        correct = 0
        model.train()

        for i, (x_train, y_train) in enumerate(train_loader):
            if torch.cuda.is_available():
                x_train = x_train.to(device)
                y_train = y_train.to(device)
            output = model(x_train)
            loss = criterion(output, y_train)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            cost_train += loss.item()

        with torch.no_grad():
            cost_valid = 0
            model.eval()
            for x_valid, y_valid in validation_loader:
                if torch.cuda.is_available():
                    x_valid = x_valid.cuda()
                    y_valid = y_valid.cuda()
                output = model(x_valid)
                loss = criterion(output, y_valid)
                _, yhat = torch.max(output, 1)
                total += y_valid.size(0)
                correct += (yhat == y_valid).sum().item()
                cost_valid += loss.item()

            scheduler.step()
            print('Valid accuracy: ', (100 * (correct / total)))
            print(f"Train. Cost: {cost_train}. Length: {len(train_loader)}. Avg: {cost_train / len(train_loader)}")
            train_loss.append(cost_train / len(train_loader))
            valid_loss.append(cost_valid / len(validation_loader))

        print(f"Train Loss: {train_loss[-1]}")
        print(f"Valid Loss: {valid_loss[-1]}")

    torch.save(model.state_dict(), "MASK_DET_model.pt")

    plt.plot(train_loss, label='Training loss')
    plt.plot(valid_loss, label='Validation loss')
    plt.legend(frameon=False)
    plt.show()


if __name__ == "__main__":
    epochs = 700
    data_dir = "/Mask Detection"
    # data_dir = "/home/abdalla/Omar/mask_detection"
    train(data_dir, epochs)
    os.environ["KMP_DUPLICATE_LIB_OK"] = "True"
