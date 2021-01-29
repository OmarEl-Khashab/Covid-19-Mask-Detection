import os
from PIL import Image
from torch.utils.data import Dataset
import torch


class MASK(Dataset):
    def __init__(self, data_dir, file_name, transform=None):
        self.transforms = transform
        self.state = ["without_mask", "with_mask"]
        self.images = []
        self.labels = []

        file_path = os.path.join(data_dir, file_name)
        # print(f"file_path: {file_path}")

        for image_path in (os.listdir(file_path)):
            # print(f"image_file: {image_path}")

            path = os.path.join(file_path, image_path)
            # print(f"path_images: {path}")

            for f in os.listdir(path):
                print(f"the image: {f}")
                self.images.append(os.path.join(path, f))
                self.labels.append(self.state.index(image_path))

    def __getitem__(self, index):
        img = self.images[index]

        label = self.labels[index]
        print(label)

        img = Image.open(img).convert(mode='RGB')

        target = torch.tensor(label, dtype=torch.float32)

        if self.transforms is not None:
            img = self.transforms(img)

        return img, target.long()

    def __len__(self):
        return len(self.images)
