import os
import torch
from torch.utils.data import DataLoader
from torchvision import transforms, datasets
from torchvision.utils import save_image
from vae import ConvVAE


def main():
    os.makedirs("./viz", exist_ok=True)
    device='mps' if torch.backends.mps.is_available() else 'cpu'
    model=ConvVAE().to(device)
    model.load_state_dict(torch.load("./checkpoint/vae.pth",map_location=device))
    model.eval()

    z_dim=model.fc_mu.out_features

    tf=transforms.Compose([
        transforms.CenterCrop(148),
        transforms.Resize(64),
        transforms.ToTensor(),
    ])
    ds=datasets.ImageFolder("./celeba_root",transform=tf)
    loader=DataLoader(ds, batch_size=8,shuffle=True,drop_last=True)

    # one fixed batch — viz 1 reconstructs these, viz 3 picks the first two
    x,_=next(iter(loader))
    x=x.to(device)

    with torch.no_grad():
        x_hat,_,_=model(x)
        recon=torch.sigmoid(x_hat)
        save_image(torch.cat([x,recon],dim=0),"./viz/reconstructions.png",nrow=8)
        z=torch.randn(64,z_dim, device=device)
        samples = torch.sigmoid(model.decode(z))
        save_image(samples, "./viz/prior_samples.png", nrow=8)
        mu,_=model.encode(x[:2])                              # (2, z_dim)
        a=torch.linspace(0, 1, 10, device=device).view(-1, 1)  # (10, 1)
        zs=(1 - a)* mu[0]+a*mu[1]                         # (10, z_dim)
        interps=torch.sigmoid(model.decode(zs))
        save_image(interps,"./viz/interpolation.png",nrow=10)  # one row, 10 frames

    print("saved: ./viz/reconstructions.png  prior_samples.png  interpolation.png")


if __name__ == "__main__":
    main()