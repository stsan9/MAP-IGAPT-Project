import torch
from torch import nn
import torch.nn.functional as F
import layers
from run_utils import convert_mask

class Generator(nn.Module):
    def __init__(self, settings):
        super(Generator, self).__init__()
        self.residual = settings["residual"]
        self.num_particles = settings["num_particles"]
        embed_dim = settings["embed_dim"]
        
        self.noise_net = layers.LinearNet(
            layers=[], input_size=settings["noise_dim"], output_size=embed_dim
        )
        
        # self.global_net = layers.LinearNet(
        #     layers=[], input_size=settings["global_noise_dim"], output_size=settings["global_feat_dim"]
        # )
        self.global_net = layers.LinearNet(
            layers=[], input_size=settings["global_noise_dim"], output_size=embed_dim
        )
        
        self.ipabs = nn.ModuleList()
        
        for _  in range(settings["ipab_layers_gen"]):
            self.ipabs.append(layers.IPAB(settings))
        
        self.output_fc = layers.LinearNet(
            layers = [],
            input_size = embed_dim,
            output_size = settings["gen_out_dim"],
            final_linear= True,
        )
    
    def forward(self, x, labels, z):

        num_jet_particles = (labels[:, -1] * self.num_particles).int() - 1

        mask = (
            (x[:, :, 0].argsort(1).argsort(1) <= num_jet_particles.unsqueeze(1))
            .unsqueeze(2)
            .float()
        )
        
        x = self.noise_net(x)
        
        z = self.global_net(z)
        
        for ipab in self.ipabs:
            sab_out, z = ipab(x, convert_mask(mask), z)
            x = x + sab_out if self.residual else sab_out
        
        x = torch.tanh(self.output_fc(x))
        
        return torch.cat((x, mask - 0.5), dim=2)
    
class Discriminator(nn.Module):
    def __init__(self, settings, residual = False):
            super(Discriminator, self).__init__()
            self.residual = residual
            embed_dim = settings["embed_dim"]
            
            self.cond_net = layers.LinearNet(
                layers = [], input_size = 2*embed_dim, output_size = embed_dim
            )
            
            self.input_net = layers.LinearNet(
                layers=[], input_size=settings["gen_out_dim"], output_size=embed_dim
            )

            self.ipabs = nn.ModuleList()
            
            for _ in range(settings["ipab_layers_disc"]):
                self.ipabs.append(layers.IPAB(settings))
            
            self.pma = layers.PMA(settings, num_seeds=1)
            
            self.output_fc = layers.LinearNet(
                layers = [],
                input_size = embed_dim,
                output_size = 1,
                final_linear= True
            )
    
    def forward(self, x):
        mask = x[..., -1:] + 0.5
        x = x[..., :-1]
        
        x = self.input_net(x)
        
        z = torch.cat([x.mean(dim=1), x.sum(dim=1)], dim=1)
        
        z = self.cond_net(z)
        
        for ipab in self.ipabs:
            sab_out, z = ipab(x, convert_mask(mask), z)
            x = x + sab_out if self.residual else sab_out
        
        out = self.pma(x, convert_mask(mask)).squeeze()
        
        return torch.sigmoid(self.output_fc(out))
