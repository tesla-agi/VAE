from torchvision import transforms,datasets
from torchvision.utils import save_image
from torch.utils.data import DataLoader
from vae import ConvVAE
import torch
import os


def train_vae():
    os.makedirs(os.path.dirname("checkpoint/vae.pth"), exist_ok=True)

    tf=transforms.Compose([
        transforms.CenterCrop(148),
        transforms.Resize(64),
        transforms.ToTensor(),
    ])

    ds = datasets.ImageFolder("./celeba_root", transform=tf)
    dataloader=DataLoader(ds,batch_size=128,shuffle=True,num_workers=0,drop_last=True)

    device='mps' if torch.backends.mps.is_available() else 'cpu'
    model=ConvVAE().to(device)
    optimizer=torch.optim.Adam(model.parameters(),lr=3e-4)

    for epoch in range(10):
        model.train()
        running=0.0
        for x,_ in dataloader:
            x=x.to(device)
            optimizer.zero_grad()
            x_hat,mu,log_var=model(x)
            loss=model.vae_loss(x,x_hat,mu,log_var)
            loss.backward()
            optimizer.step()
            running=running+loss.item()
        print(f"epoch {epoch}: {running / len(dataloader):.1f}")

    torch.save(model.state_dict(),"./checkpoint/vae.pth")
    print("Model Saved")


if "__main__" == __name__:
    train_vae()
