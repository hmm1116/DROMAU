import csv
import os
import os.path as osp

from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
from torchvision.transforms import InterpolationMode


SPLIT_PATH = "./data/RFMiD/split"


def _read_split(csv_path):
    rows = []
    with open(csv_path, "r", newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append((row["source"], row["image"], row["label"]))
    return rows


def _image_candidates(root, source, image_id):
    names = [image_id]
    if not osp.splitext(image_id)[1]:
        names = [image_id + ext for ext in (".png", ".jpg", ".jpeg")]

    if source == "session":
        folders = [
            "Training",
            "train",
            "",
        ]
    elif source == "test":
        folders = [
            "Test",
            "test",
            "",
        ]
    else:
        folders = [""]

    for folder in folders:
        for name in names:
            yield osp.join(root, folder, name)


class RFMiD(Dataset):
    def __init__(self, setname, args):
        split_name = "fsl_train.csv" if setname == "train" else "fsl_test.csv"
        split_path = getattr(args, "rfmid_split_path", None)
        split_path = split_path or os.environ.get("RFMID_SPLIT_PATH") or SPLIT_PATH
        csv_path = osp.join(split_path, split_name)
        rows = _read_split(csv_path)

        image_root = getattr(args, "rfmid_image_path", None)
        image_root = image_root or os.environ.get("RFMID_IMAGE_PATH")
        if not image_root:
            raise ValueError("Set --rfmid_image_path or RFMID_IMAGE_PATH")

        data = []
        label = []
        lb = -1
        self.wnids = []

        for source, image_id, wnid in rows:
            if wnid not in self.wnids:
                self.wnids.append(wnid)
                lb += 1

            candidates = list(_image_candidates(image_root, source, image_id))
            img_path = next((path for path in candidates if osp.exists(path)), candidates[0])
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
                transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
                transforms.RandomHorizontalFlip(),
                transforms.RandomRotation(90),
                transforms.ToTensor(),
                transforms.Normalize(normalize_mean, normalize_std),
            ])
        else:
            self.transform = transforms.Compose([
                transforms.Resize(input_size, interpolation=interpolation),
                transforms.CenterCrop(input_size),
                transforms.ToTensor(),
                transforms.Normalize(normalize_mean, normalize_std),
            ])

    def __len__(self):
        return len(self.data)

    def __getitem__(self, i):
        path, label = self.data[i], self.label[i]
        image = self.transform(Image.open(path).convert("RGB"))
        return image, label
