# save_celeba_sample.py
import os, torch
from torch.utils.data import DataLoader
from torchvision import transforms, datasets
from torchvision.utils import save_image

os.makedirs("./assets", exist_ok=True)
tf = transforms.Compose([
    transforms.CenterCrop(148),
    transforms.Resize(64),
    transforms.ToTensor(),
])
ds = datasets.ImageFolder("./celeba_root", transform=tf)
loader = DataLoader(ds, batch_size=64, shuffle=True)
x, _ = next(iter(loader))
save_image(x, "./assets/celeba_sample.png", nrow=8)
print("saved ./assets/celeba_sample.png")