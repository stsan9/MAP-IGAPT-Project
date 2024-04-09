import math
import torch
from torch import nn
import torch.nn.functional as F

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
    def __init__(self, embed_dim, num_heads, activation_function = torch.softmax, layer_norm = True, dropout = 0.1):
        assert embed_dim % num_heads == 0, "Embedding dimension must be divisible by number of heads"
        
        super(MAB, self).__init__()
        self.activation_function = activation_function
        
        self.attention = nn.MultiheadAttention(embed_dim=embed_dim, num_heads=num_heads, dropout=dropout)
        self.layer_norm = nn.LayerNorm(embed_dim) if layer_norm else None
        
        self.feedforward = nn.LinearNet() # FIXME: THIS IS A PLACEHOLDER FOR THE FEEDFORWARD LAYER
    
    def forward(self, X, Y):
        attention = self.attention(X, Y, Y)
        
        if self.layer_norm:
            H = self.layer_norm(X + attention)
            
        mab_out = H + self.feedforward(H)
        
        if self.layer_norm:
            return self.layer_norm(mab_out)
        
        return mab_out
    
class PMA(nn.Module):
    def __init__(self, embed_dims, seed_count, **mab_args):
        super(PMA, self).__init__()
        self.seed_count = seed_count
        
        self.feedforward = nn.LinearNet() # FIXME: THIS IS A PLACEHOLDER FOR THE FEEDFORWARD LAYER
        self.mab = MAB(embed_dims, **mab_args)
        self.S = nn.Parameter(torch.empty(1, seed_count, embed_dims))
        nn.init.xavier_uniform_(self.S)
    
    def forward(self, X):
        return self.mab(self.S.repeat(X.size(0), 1, 1), X)

class ISAB(nn.Module):
    def __init__(self, m_induce, embed_dim, **mab_args):
        super(ISAB, self).__init__()
        
        self.m = m_induce
        self.mab1 = MAB(embed_dim, **mab_args)
        self.mab2 = MAB(embed_dim, **mab_args)
        self.I = nn.Parameter(torch.randn(self.m, embed_dim))
    
    def forward(self, X):
        H = self.mab1(self.I, X)
        return self.mab2(X, H)

class IPAB(nn.Module):
    def __init__(self, embed_dim, **mab_args):
        super(IPAB, self).__init__()
        self.embed_dim = embed_dim
        
        self.mab1 = MAB(embed_dim, **mab_args) 
        self.mab2 = MAB(embed_dim, **mab_args)
    
    def forward(self, X, Z):
        Z = self.mab1(Z.unsqueeze(1), X)
        return self.mab2(X, Z), Z.squeeze(1)

