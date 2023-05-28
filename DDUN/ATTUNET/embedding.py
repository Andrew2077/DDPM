import torch

def pos_encoding(t, dim):
    # * range from 0 to dim of embedding, step = 2
    freq_range = torch.arange(0, dim, 2, ).float().to(t.device)
    # * divide by dim of embedding
    normalized_range = freq_range / dim
    # * 1 / 10000 ^ normalized_range -> idea behind positional encoding
    inv_freq = 1.0 / (10000**normalized_range)

    #*  unsqueeze t to shape (batch_size, 1)
    if shape := t.shape:
        if len(shape) == 1:
            t = t.unsqueeze(-1)
    # * repeat t for each dim of embedding - shape (len(t), dim//2)
    t = t.repeat(1, dim // 2)
    
    #* calculate positional encoding
    pos_enc_a = torch.sin(t * inv_freq)
    pos_enc_b = torch.cos(t * inv_freq)
    
    #* concat columns
    pos_enc = torch.cat([pos_enc_a, pos_enc_b], dim=-1)
    return pos_enc