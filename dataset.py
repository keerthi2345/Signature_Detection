import os
import random
from PIL import Image
from torch.utils.data import Dataset, WeightedRandomSampler
from torchvision import transforms
import torch
from collections import Counter

IMAGE_SIZE = (128, 128)
train_transforms = transforms.Compose([
    transforms.RandomResizedCrop(IMAGE_SIZE, scale=(0.8, 1.0)),
    transforms.RandomRotation(6),
    transforms.RandomAffine(0, shear=5),
    transforms.ColorJitter(brightness=0.2, contrast=0.2),
    transforms.ToTensor(),
    transforms.Normalize((0.5,0.5,0.5), (0.5,0.5,0.5))
])

eval_transforms = transforms.Compose([
    transforms.Resize(IMAGE_SIZE),
    transforms.ToTensor(),
    transforms.Normalize((0.5,0.5,0.5), (0.5,0.5,0.5))
])

class SignatureDataset(Dataset):
    """
    Expects processed images in:
      data/processed/positive/   (all signature images)
      data/processed/negative/   (all non-signature images)
    Images can be RGB.
    """
    def __init__(self, root_dir="data/processed", mode="train"):
        assert mode in ("train","eval")
        self.pos_dir = os.path.join(root_dir, "positive")
        self.neg_dir = os.path.join(root_dir, "negative")

        self.images = []
        self.labels = []

        for fn in os.listdir(self.pos_dir):
            if fn.lower().endswith((".png",".jpg",".jpeg")):
                self.images.append(os.path.join(self.pos_dir, fn))
                self.labels.append(1)

        for fn in os.listdir(self.neg_dir):
            if fn.lower().endswith((".png",".jpg",".jpeg")):
                self.images.append(os.path.join(self.neg_dir, fn))
                self.labels.append(0)

        if mode == "train":
            self.transform = train_transforms
        else:
            self.transform = eval_transforms

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        path = self.images[idx]
        label = self.labels[idx]
        img = Image.open(path).convert("RGB")
        img = self.transform(img)
        return img, torch.tensor(label, dtype=torch.float32)

def get_weighted_sampler(dataset):
    """
    Returns a WeightedRandomSampler that balances classes in each epoch.
    """
    counts = Counter(dataset.labels)
    class_sample_count = [counts[0], counts[1]]
    # inverse frequency weights
    weight_per_class = [0., 0.]
    for i, c in enumerate(class_sample_count):
        weight_per_class[i] = 1.0 / (c + 1e-6)
    weights = [weight_per_class[int(l)] for l in dataset.labels]
    sampler = WeightedRandomSampler(weights, num_samples=len(weights), replacement=True)
    return sampler
