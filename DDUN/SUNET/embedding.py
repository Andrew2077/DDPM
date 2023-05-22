import torch 
import math

class SinsuoidalPostionalEmbedding(torch.nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim
        
    def forward(self, sequence, device = None):
        """
        equation: 
            - n  =log_2(10000) / (d - 1)
            - d = dim // 2
            - embedding = exp(d * -n) for d = 0, 1, 2, ..., d-1
            - embedding_space = sequence[:, None] * embedding[None, :]
            - sine_embedding = sin(embedding_space)
            - cosine_embedding = cos(embedding_space)
            - embedding = concat(sine_embedding, cosine_embedding)
        """
        #print(device)
        d = self.dim // 2
        n = math.log(10000) / (d - 1 + 1e-8) #* n
        
        #* d^(-n) where d = 0, 1, 2, ..., d-1
        #* tensor of shape (d,)
        embedding = torch.exp(torch.arange(d, device=device) * -n).to(device)
        embedding = sequence[:, None].to(device) * embedding[None, :].to(device) #* (seq_len, d)
        
        #* sin(embedding) and cos(embedding), then concat them
        embedding = torch.cat((embedding.sin(), embedding.cos()), dim=-1) #* (seq_len, 2d)
        embedding = embedding.to(device)
        return embedding