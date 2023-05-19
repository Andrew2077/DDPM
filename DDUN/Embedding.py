from torch import nn
import torch
import math
class SinsoidalPositionalEmbeddings(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim
        
    def forward(self, timestep): 
        #* get the device of the timestep
        device = timestep.device 
        
        #* get the half of the dimension
        half_dim = self.dim // 2 
        
        #* calculate the frequency #* 2^i / 10000^(2i/d)
        embeddings = math.log(10000) / (half_dim - 1+1e-6) #* 1e-6 is added to avoid division by zero
        
        #* calculate the embeddings
        embeddings = torch.exp(torch.arange(half_dim, device=device) * -embeddings) 
        embeddings = timestep[:, None] * embeddings[None, :]
        embeddings = torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)
        
        return embeddings

        