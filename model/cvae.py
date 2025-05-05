
import torch
import torch.nn as nn

class CVAE(nn.Module):
    def __init__(self, cond_dim, item_dim, hidden_dim=128, latent_dim=64):
        super(CVAE, self).__init__()
        self.cond_dim = cond_dim
        self.item_dim = item_dim
        self.encoder = nn.Sequential(
            nn.Linear(cond_dim + item_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )
        self.fc_mu = nn.Linear(hidden_dim, latent_dim)
        self.fc_logvar = nn.Linear(hidden_dim, latent_dim)

        self.decoder = nn.Sequential(
            nn.Linear(latent_dim + cond_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, item_dim),
            nn.Sigmoid()
        )

    def encode(self, cond, item):
        x = torch.cat([cond, item], dim=1)
        h = self.encoder(x)
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        return mu, logvar

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z, cond):
        x = torch.cat([z, cond], dim=1)
        return self.decoder(x)

    def forward(self, cond, item):
        mu, logvar = self.encode(cond, item)
        z = self.reparameterize(mu, logvar)
        recon = self.decode(z, cond)
        return recon, mu, logvar
