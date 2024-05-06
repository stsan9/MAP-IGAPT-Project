import math
import torch
from torch import nn, Tensor
import torch.nn.functional as F


class LinearNet(nn.Module):
    def __init__(self, input_size, output_size, layers=[], leaky_relu_alpha = 0.2, dropout_p = 0, final_linear = False):
        super(LinearNet, self).__init__()
        
        layers = [input_size] + layers + [output_size]
        self.final_linear = final_linear
        self.leaky_relu_alpha = leaky_relu_alpha
        self.dropout = nn.Dropout(p=dropout_p)
        
        self.net = nn.ModuleList()
        for i in range(len(layers) - 1):
            linear = nn.Linear(layers[i], layers[i + 1], device = "cuda")
            
            self.net.append(linear)
        
    def forward(self, x):
        for i in range(len(self.net)):
            x = self.net[i](x)
            if i != len(self.net) - 1 or not self.final_linear:
                x = F.leaky_relu(x, negative_slope=self.leaky_relu_alpha)
                
            x = self.dropout(x)
        
        return x        

# Kind of useless since it would be slow to loop this but it's a good example of how to implement it
class Attention(nn.Module):
    def __init__(self, activation_function = torch.softmax):
        super(Attention, self).__init__()
        self.activation_function = activation_function
    
    def forward(self, Q, K, V):
        d = Q.size(-1)
        logits = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(d)
        logits = self.activation_function(logits)
        return torch.matmul(logits, V)

class MAB(nn.Module):
    def __init__(self, settings, ff_layers = [], final_linear = True, activation_function = torch.softmax, layer_norm = False, dropout = 0.1):
        assert settings["embed_dim"] % settings["num_heads"] == 0, "Embedding dimension must be divisible by number of heads"
        
        super(MAB, self).__init__()
        embed_dim = settings["embed_dim"]
        self.num_heads = settings["num_heads"]
        self.activation_function = activation_function
        
        self.attention = nn.MultiheadAttention(embed_dim=embed_dim, num_heads=self.num_heads, batch_first=True, device="cuda")
        self.layer_norm = nn.LayerNorm(embed_dim) if layer_norm else None
        
        self.feedforward = LinearNet(embed_dim, embed_dim, ff_layers, final_linear=final_linear) 
        self.dropout = nn.Dropout(p=dropout)
    
    def forward(self, X, Y, y_mask: Tensor = None):



        
        if y_mask is not None:
            # torch.nn.MultiheadAttention needs a mask of shape [batch_size * num_heads, N, N]
            y_mask = torch.repeat_interleave(y_mask, self.num_heads, dim=0)



        
        
        attention = self.attention(X, Y, Y, y_mask, need_weights=False)[0]

        
        H = X + attention
        
        if self.layer_norm:
            H = self.layer_norm(H)
        H = self.dropout(H)
        
        mab_out = H + self.feedforward(H)
        
        if self.layer_norm:
            mab_out = self.layer_norm(mab_out)
        mab_out = self.dropout(mab_out)
        
        return mab_out
    
class PMA(nn.Module):
    def __init__(self, settings, num_seeds):
        super(PMA, self).__init__()
        self.seed_count = num_seeds
        
        self.mab = MAB(settings)
        self.S = nn.Parameter(torch.empty(1, num_seeds, settings["embed_dim"]))
        nn.init.xavier_uniform_(self.S)
    
    def forward(self, X, mask):
        mask = mask.transpose(-2, -1)
        return self.mab(self.S.repeat(X.size(0), 1, 1).to(X.device), X)

class ISAB(nn.Module):
    def __init__(self, m_induce, embed_dim, **mab_args):
        super(ISAB, self).__init__()
        
        self.m = m_induce
        self.mab1 = MAB(embed_dim, **mab_args)
        self.mab2 = MAB(embed_dim, **mab_args)
        self.I = nn.Parameter(torch.randn(self.m, embed_dim))
    
    def forward(self, X):
        if mask is not None:
            mask = mask.transpose(-2, -1).repeat((1, self.num_inds, 1))
        H = self.mab1(self.I, X)
        return self.mab2(X, H)

class IPAB(nn.Module):
    def __init__(self, settings):
        super(IPAB, self).__init__()
        self.embed_dim = settings["embed_dim"]
        
        self.mab1 = MAB(settings) 
        self.mab2 = MAB(settings)
    
    def forward(self, X, mask, Z):
        if mask is not None:
            mask = mask.transpose(-2, -1).repeat((1, self.num_inds, 1)) #Follow up with num_inds
        Z = self.mab1(Z.unsqueeze(1), X)
        return self.mab2(X, Z), Z.squeeze(1)

if __name__ == "__main__":
    # Test the layers
    net = LinearNet([10, 20, 30, 1])
    x = torch.randn(16,10)
    # print(net(x), net(x).shape)
    
    torch.manual_seed(0)
    ipab = IPAB(10, num_heads=2, ff_layers = [10,20,10])
    x = torch.ones(16, 30, 10)
    z = torch.ones(16, 10)
    print(ipab(x, z))
    
