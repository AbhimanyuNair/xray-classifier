
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image

LABEL2ID = {"NORMAL": 0, "PNEUMONIA": 1}
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

class XrayDataset(Dataset):
    def __init__(self, df, img_dir, transform=None):
        self.df = df.reset_index(drop=True)
        self.img_dir = img_dir
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, i):
        row = self.df.iloc[i]
        img = Image.open(self.img_dir / row.cache_name).convert("RGB")
        if self.transform:
            img = self.transform(img)
        return img, int(row.target)

def get_transforms(img_size=224):
    norm = transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD)
    train_tf = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.RandomAffine(degrees=10, translate=(0.05, 0.05), scale=(0.95, 1.05)),
        transforms.ColorJitter(brightness=0.15, contrast=0.15),
        transforms.ToTensor(),
        norm,
    ])
    eval_tf = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        norm,
    ])
    return train_tf, eval_tf

def make_loaders(splits_df, img_dir, img_size=224, batch_size=32, num_workers=2, seed=42):
    train_tf, eval_tf = get_transforms(img_size)
    g = torch.Generator().manual_seed(seed)
    loaders = {}
    for part in ["train", "val", "test"]:
        d = splits_df[splits_df.partition == part]
        ds = XrayDataset(d, img_dir, train_tf if part == "train" else eval_tf)
        loaders[part] = DataLoader(
            ds, batch_size=batch_size, shuffle=(part == "train"),
            num_workers=num_workers, pin_memory=True,
            generator=g if part == "train" else None,
        )
    return loaders
