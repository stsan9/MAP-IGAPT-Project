import torch
from torch import nn
import torch.nn.functional as F
import layers

class Generator(nn.Module):
    def __init__(self, settings):
        super(Generator, self).__init__()
        embed_dim = settings["embed_dim"]
        
        self.noise_net = layers.LinearNet(
            layers=[], input_size=settings["noise_dim"], output_size=embed_dim
        )
        
        self.global_net = layers.LinearNet(
            layers=[], input_size=settings["global_noise_dim"], output_size=settings["global_feat_dim"]
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
    
    def forward(self, x, mask, z):
        # TODO: implement mask
        x = self.noise_net(x)
        
        z = self.global_net(z)
        
        for ipab in self.ipabs:
            sab_out, z = ipab(x, mask, z)
            x = x + sab_out if self.residual else sab_out
        
        x = torch.tanh(self.output_fc(x))
        
        return x
    
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
    
    def forward(self, x, mask):
        x = self.input_net(x)
        
        z = torch.cat([x.mean(dim=1), x.sum(dim=1)], dim=1)
        
        z = self.cond_net(z)
        
        for ipab in self.ipabs:
            sab_out, z = ipab(x, mask, z)
            x = x + sab_out if self.residual else sab_out
        
        out = self.pma(x, mask, z).squeeze()
        
        return torch.sigmoid(self.output_fc(out))
