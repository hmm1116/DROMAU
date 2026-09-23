import os
import os.path as osp

import numpy as np
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
from torchvision.transforms import InterpolationMode


SPLIT_PATH = "./data/isic/split"


def loadtest(csv_path):
    paths = csv_path
    paths = np.loadtxt(paths, delimiter=",", skiprows=0, dtype=str)
    paths = paths[1:]
    paths = np.core.defchararray.replace(paths, '"', '', count=None)
    count = len(paths)
    path = []
    labelx = []
    imgs = []
    for i in range(count):
        path.append(paths[i][0])
        labelx.append(np.argmax(paths[i][1:]))
        imgs.append((paths[i][0], np.argmax(paths[i][1:])))
    return imgs


class ISIC(Dataset):
    def __init__(self, setname, args):
        split_path = getattr(args, "isic_split_path", None)
        split_path = split_path or os.environ.get("ISIC_SPLIT_PATH") or SPLIT_PATH
        csv_path = osp.join(split_path, setname + ".csv")
        super().__init__()
        self.namelabel = loadtest(csv_path)
        image_path = getattr(args, "isic_image_path", None)
        self.train_path = image_path or os.environ.get("ISIC_IMAGE_PATH")
        if not self.train_path:
            raise ValueError("Set --isic_image_path or ISIC_IMAGE_PATH")
        data = []
        label = []
        lb = -1

        self.wnids = []

        for i in range(len(self.namelabel)):
            img_name, labelx = self.namelabel[i]
            if labelx not in self.wnids:
                self.wnids.append(labelx)
                lb += 1
            img_name = "".join(img_name)
            img_name += '.jpg'
            img_path = osp.join(self.train_path, img_name)
            data.append(img_path)
            label.append(lb)

        self.data = data
        self.label = label
        self.num_class = len(set(label))

        input_size = 224
        if getattr(args, "model", None) == "medical_vit":
            normalize_mean = [0.48145466, 0.4578275, 0.40821073]
            normalize_std = [0.26862954, 0.26130258, 0.27577711]
            interpolation = InterpolationMode.BICUBIC
        else:
            normalize_mean = [0.485, 0.456, 0.406]
            normalize_std = [0.229, 0.224, 0.225]
            interpolation = InterpolationMode.BILINEAR
        if setname == "train":
            self.transform = transforms.Compose([
                transforms.Resize(input_size, interpolation=interpolation),
                transforms.CenterCrop(input_size),
                transforms.RandomCrop(input_size, padding=4),
                transforms.RandomAffine(degrees=0, translate=(0.2, 0.2)),
                transforms.RandomHorizontalFlip(),
                transforms.RandomRotation(90),
                transforms.ToTensor(),
                transforms.Normalize(normalize_mean, normalize_std)
            ])

        else:
            self.transform = transforms.Compose([
                transforms.Resize(input_size, interpolation=interpolation),
                transforms.CenterCrop(input_size),
                transforms.ToTensor(),
                transforms.Normalize(normalize_mean, normalize_std)
            ])

    def __len__(self):
        return len(self.namelabel)

    def __getitem__(self, i):
        path, label = self.data[i], self.label[i]
        image = self.transform(Image.open(path).convert("RGB"))

        return image, label
