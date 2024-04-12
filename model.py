import torch
from torch import nn
import torch.nn.functional as F
from . import layers, preprocessing

class Generator(nn.Module):
    def __init__(self, 
                 particle_count,
                 output_dim, 
                 embed_dim = 32,
                 noise_dim = 8,
                 global_noise_dim = 0,
                 global_feat_dim = 0,
                 num_ipabs = 2,
                 learnable_noise = False, 
                 residual = False,
                 global_net_layers = [],
                 output_fc_layers = [],
                 **mab_args):
        super(Generator, self).__init__()
        
        if learnable_noise:
           self.mu = nn.Parameter(torch.randn(particle_count , embed_dim))
           self.std = nn.Parameter(torch.randn(particle_count, embed_dim))
        
        self.noise_net = layers.LinearNet(
            layers=[], input_size=noise_dim, output_size=embed_dim
        )
        
        self.global_net = layers.LinearNet(
            layers=global_net_layers, input_size=global_noise_dim, output_size=embed_dim
        )
        
        self.ipabs = nn.ModuleList()
        
        for _  in range(num_ipabs):
            self.ipabs.append(layers.IPAB(embed_dim, **mab_args))
        
        self.output_fc = layers.LinearNet(
            output_fc_layers,
            input_size = embed_dim,
            output_size = output_dim,
            activation_function = lambda x: x,
            layer_norm = False
        )
    
    def forward(self, x, mask, z):
        # TODO: implement mask
        x = self.noise_net(x)
        
        z = self.global_net(z)
        
        for ipab in self.ipabs:
            sab_out, z = ipab(x, mask, z)
            x = x + sab_out if self.residual else sab_out
        
        x = torch.tanh(self.output_fc(x))
        
        return x
    
    def sample_noise(self, batch_size):
        cov = torch.eye(self.std.shape[1]).repeat(self.num_particles, 1, 1).to(
                self.std.device
            ) * (self.std**2).unsqueeze(2)
        
        mvn = torch.distributions.MultiVariateNormal(self.mu, cov)
        
        return mvn.rsample((batch_size,))

class Discriminator(nn.Module):
    def __init__(self,
                    particle_count,
                    input_dim,
                    embed_dim = 32,
                    cond_feat_dim = 8,
                    global_feat_dim = 0,
                    num_ipabs = 2,
                    global_net_layers = [],
                    cond_net_layers = [],
                    output_fc_layers = [],
                    residual = False,
                    **mab_args):
            super(Discriminator, self).__init__()
            self.residual = residual
            
            self.cond_net = layers.LinearNet(
                layers = cond_net_layers, input_size = 2*embed_dim, output_size = embed_dim
            )
            
            self.input_net = layers.LinearNet(
                layers=[], input_size=input_dim, output_size=embed_dim
            )

            self.ipabs = nn.ModuleList()
            
            for _ in range(num_ipabs):
                self.ipabs.append(layers.IPAB(embed_dim, **mab_args))
            
            self.pma = layers.PMA(num_seeds=1, embed_dim=embed_dim, **mab_args)
            
            self.output_fc = layers.LinearNet(
                output_fc_layers,
                input_size = embed_dim,
                output_size = 1,
                activation_function = lambda x: x,
                layer_norm = False
            )
    
    def forward(self, x, mask):
        x = self.input_net(x)
        
        z = torch.cat([x.mean(dim=1), x.sum(dim=1)], dim=1)
        
        z = self.cond_net(z)
        
        for ipab in self.ipabs:
            sab_out, z = ipab(x, mask, z)
            x = x + sab_out if self.residual else sab_out
        
        out = self.pma(x, mask, z).squeeze()
        
        return torch.sigmoid(self.output_fc(out))