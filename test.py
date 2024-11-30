from torch.utils.data import DataLoader
from torchvision.transforms import RandomRotation
from model import Resnet34
from mask_dataset import MASK
import torch
from torchvision import transforms
from sklearn.metrics import confusion_matrix, f1_score
import matplotlib.pyplot as plt
import os


def test(data_dir):
    test_transforms = transforms.Compose(
        [
            transforms.Resize(size=100),
            transforms.CenterCrop(size=50),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ]
    )

    test_dataset = MASK(data_dir, "test", test_transforms)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=64, shuffle=True, num_workers=4)

    # load the model
    model = Resnet34()
    model.load_state_dict(torch.load("MASK_DET_model.pt", map_location=torch.device('cpu')))

    if torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device("cpu")

    model = model.to(device)

    with torch.no_grad():

        model.eval()
        total = 0
        correct = 0
        pred=None
        true=None
        for x_test, y_test in test_loader:
            if torch.cuda.is_available():
                x_test = x_test.to(device)
                y_test = y_test.to(device)

            output = model(x_test)

            _, yhat = torch.max(output, 1)
            total += y_test.size(0)
            correct += (yhat == y_test).sum().item()
            if pred is None:
                pred = yhat.unsqueeze(1)
            else:
                pred = torch.cat([pred, yhat.unsqueeze(1)])
            if true is None:
                true = y_test.unsqueeze(1)
            else:
                true = torch.cat([true, y_test.unsqueeze(1)])
        print('Test accuracy: ', (100 * (correct / total)))
        f1 = f1_score(true.cpu().numpy(), pred.cpu().numpy(), average=None)
        print(f"f1_score: {f1}")


if __name__ == "__main__":
    data_dir = "/Mask Detection"
    # data_dir = "/home/abdalla/Omar/mask_detection"
    test(data_dir)
