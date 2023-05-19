import torch.nn.functional as F
import torch

def intiate_betas(T, START, END):
    return torch.linspace(START, END, T)

#* cumprod: , alphas, alphas_pred
def calc_cumprod_alphas(betas):
    return torch.cumprod(1 - betas, dim=0)

def clac_cumoprod_alphas_prev(betas):
    Alphas = 1 - betas
    return F.pad(Alphas[:-1], (1, 0), value=1)


#* sqrt_recip: alphas_recip, alphas_cumpord, alphas_one_minus_cumpord
def calc_sqrt_recip_alpha(betas):
    return torch.sqrt(1 / (1 - betas))

def calc_sqrt_cumprod_alphas(betas):
    return torch.sqrt(torch.cumprod(1 - betas, dim=0))

def calc_sqrt_one_minus_cumpord_alphas(betas):
    return torch.sqrt(1 - torch.cumprod(1 - betas, dim=0))

#* postierior_variance:
def calc_postierior_variance(betas):
    return betas * (1 - clac_cumoprod_alphas_prev(betas)) / (1 - calc_cumprod_alphas(betas))


#* Forward
def get_index_from_list(list, t, image_shape):
    batch_size = t.shape[0] #* get the batch size
    out = list.gather(-1, t.cpu()) #* gather the values from the list & move to cpu
    #* reshape the output to match the shape of the input.
    out = out.reshape(batch_size, *((1,)) * (len(image_shape) - 1)).to(t.device) 
    return out


def Forward_pass(
    image, 
    t, 
    sqrt_alphas_cumpord,
    sqrt_one_minus_alphas_cumpord,
    device = "cpu",
    torch_seed = 42
):  
    torch.manual_seed(torch_seed)
    #* initialize the noise
    noise = torch.randn_like(image)
    
    #* get the sqrt of alphas cumpord at time t
    sqrt_alphas_cumpord_t = get_index_from_list(sqrt_alphas_cumpord, t, image.shape)
    
    #* get the sqrt of one minus alphas cumpord at time t
    sqrt_one_minus_alphas_cumpord_t = get_index_from_list(sqrt_one_minus_alphas_cumpord, t, image.shape)
    
    image_part = sqrt_alphas_cumpord_t.to(device) * image.to(device)
    noise_part = sqrt_one_minus_alphas_cumpord_t.to(device) * noise.to(device)
    
    return image_part + noise_part, noise_part