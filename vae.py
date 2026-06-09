import torch
import torch.nn as nn
import torch.nn.functional as F
import sys
class ConvVAE(nn.Module):
    def __init__(self,z_dim=32):
        super(ConvVAE,self).__init__()

        self.encoder = nn.Sequential(
            nn.Conv2d(3,32,kernel_size=4,stride=2,padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32,64,kernel_size=4,stride=2,padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64,128,kernel_size=4,stride=2,padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128,256,kernel_size=4,stride=2,padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
        )
        self.fc_mu=nn.Linear(256*4*4,z_dim)
        self.fc_log_var=nn.Linear(256*4*4,z_dim)
        self.fc_dec = nn.Linear(z_dim, 256 * 4 * 4)
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(256,128,kernel_size=4,stride=2,padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.ConvTranspose2d(128,64,kernel_size=4,stride=2,padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.ConvTranspose2d(64,32,kernel_size=4,stride=2,padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.ConvTranspose2d(32,3,kernel_size=4,stride=2,padding=1),

        )

    def encode(self,x):
        x=self.encoder(x).flatten(1)
        mu=self.fc_mu(x)
        log_var=self.fc_log_var(x)
        return mu,log_var

    def reparameterize(self,mu,log_var):
        std=torch.exp(0.5*log_var)
        eps=torch.randn_like(std)
        z=mu+eps*std
        return z

    def decode(self,z):
        z=self.fc_dec(z).view(-1,256,4,4)
        x_hat=self.decoder(z)
        return x_hat

    def forward(self,x):
        mu,log_var=self.encode(x)
        z=self.reparameterize(mu,log_var)
        x_hat=self.decode(z)
        return x_hat,mu,log_var

    def vae_loss(self,x,x_hat,mu,log_var):
        recon_loss=F.binary_cross_entropy_with_logits(x_hat,x,reduction='sum')/x.size(0)
        kl=-0.5*torch.mean(torch.sum(1+log_var-mu.pow(2)-log_var.exp(), dim=1))
        total_loss=recon_loss+kl
        return total_loss




sys.exit(0)
'''

Testing Groud for the VAE ( Architecture is given above ) 
m = ConvVAE()
x = torch.randn(8, 3, 64, 64)
x_hat, mu, log_var = m(x)
print(x_hat.shape, mu.shape, log_var.shape)   # (8,3,64,64) (8,32) (8,32)

'''


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = ConvVAE().to(device)
    z_dim = model.fc_mu.out_features

    # dummy batch — pixels in [0,1] (BCE_with_logits targets must be in [0,1], so rand not randn)
    x = torch.rand(8, 3, 64, 64, device=device)

    # 1) forward — check shapes
    x_hat, mu, log_var = model(x)
    print("forward  ->", tuple(x_hat.shape), tuple(mu.shape), tuple(log_var.shape))

    # 2) loss — should be a finite scalar
    loss = model.vae_loss(x, x_hat, mu, log_var)
    print("loss     ->", round(loss.item(), 3))

    # 3) one optimization step — confirms gradients flow end to end (incl. through reparameterize)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    opt.zero_grad(); loss.backward(); opt.step()
    print("backward -> one step OK (gradients flow)")

    # 4) generation — decode a prior sample z ~ N(0, I)
    model.eval()
    with torch.no_grad():
        z = torch.randn(8, z_dim, device=device)
        samples = torch.sigmoid(model.decode(z))
    print("generate ->", tuple(samples.shape),
          f"pixels in [{samples.min().item():.2f}, {samples.max().item():.2f}]")


if __name__ == "__main__":
    main()