import os
import os.path as osp

from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
from torchvision.transforms import InterpolationMode


SPLIT_PATH = "./data/SD198/split"


class SD198(Dataset):
    def __init__(self, setname, args):
        split_path = getattr(args, "sd198_split_path", None)
        split_path = split_path or os.environ.get("SD198_SPLIT_PATH") or SPLIT_PATH
        image_path = getattr(args, "sd198_image_path", None)
        image_path = image_path or os.environ.get("SD198_IMAGE_PATH")
        if not image_path:
            raise ValueError("Set --sd198_image_path or SD198_IMAGE_PATH")
        csv_path = osp.join(split_path, setname + ".csv")
        lines = [x.strip() for x in open(csv_path, "r").readlines()][1:]

        data = []
        label = []
        lb = -1

        self.wnids = []

        for l in lines:
            name, wnid = l.split(",")
            path = osp.join(image_path, name)
            if wnid not in self.wnids:
                self.wnids.append(wnid)
                lb += 1
            data.append(path)
            label.append(lb)

        self.data = data
        self.label = label
        self.num_class = len(set(label))

        # Transformation

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
                # transforms.Resize(336),
                # transforms.RandomResizedCrop(input_size),
                transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
                transforms.RandomHorizontalFlip(),
                #transforms.RandomVerticalFlip(),
                transforms.RandomRotation(180),
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
        return len(self.data)

    def __getitem__(self, i):
        path, label = self.data[i], self.label[i]
        image = self.transform(Image.open(path).convert("RGB"))
        return image, label
