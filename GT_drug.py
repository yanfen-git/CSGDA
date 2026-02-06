import torch.nn as nn
import torch.nn.functional as F
import torch
from GTLayer import GraphTransformer

class GT(nn.Module):
    def __init__(self, nfeat, nhid, nclass, dropout):
        super(GT, self).__init__()
        self.gc1 = GraphTransformer(nfeat, nhid, num_heads=4, num_layers=3)
        self.gc2 = GraphTransformer(nhid, nclass, num_heads=4, num_layers=3)
        self.gc3 = GraphTransformer(nhid, nclass, num_heads=4, num_layers=3)
        self.dropout = dropout

    def reparameterize(self, mu, logvar):
        if self.training:
            std = torch.exp(logvar) + 1e-12
            eps = torch.randn_like(std)
            z = eps.mul(std).add_(mu)
            if torch.isnan(z).any():
                print("NaN detected in reparameterize output")
            return z
        else:
            return mu

    def forward(self, g, features):
        with g.local_scope():
            x = self.gc1(g, features)
            x = F.relu(F.normalize(x))
            x = F.dropout(x, self.dropout, training=self.training)

            mu = F.normalize(self.gc2(g, x))
            logvar = F.normalize(self.gc3(g, x))

            z = self.reparameterize(mu, logvar)
            return z