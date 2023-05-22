import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from tqdm.notebook import tqdm


class MarkovDDPM(nn.Module):
    def __init__(self, network, n_steps, start, end, device, image_ch=((1, 28, 28))):
        super().__init__()
        
        self.network = network
        self.n_steps = n_steps
        self.start = start
        self.end = end
        self.device = device
        self.image_ch = image_ch
        
        #* define process 
        # self.betas = torch.linspace(start, end, n_steps).to(device)
        # self.alphas = 1 - self.betas
        # #self.alpha_bars = torch.tensor([torch.prod(self.alphas[:i+1]) for i in range(len(self.alphas))]).to(device)
        # self.alpha_bars = torch.cumprod(self.alphas, dim=0).to(device)
        
        ################ Define the process -- HuggingFace################
        #* Alphas
        self.betas = torch.linspace(start, end, n_steps).to(device)
        self.alphas = 1 - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0).to(device)
        self.alphas_cumprod_prev = F.pad(self.alphas_cumprod[:-1], (1, 0), value=1.0)
        self.sqrt_recip_alphas = torch.sqrt(1.0 / self.alphas)
        
        #* calculation for diffusion q(x_t| x_{t-1})
        # self.alpha_bars = torch.cumprod(self.alphas, dim=0).to(device)
        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - self.alphas_cumprod)
        
        #* calculation for posterior  q(x_{t-1}| x_t, x_0)})
        self.posterior_variance = self.betas * (1. - self.alphas_cumprod_prev) / (1. - self.alphas_cumprod)
        
    def extract(self, a, t , x_shape):
        batch_size = t.shape[0]
        #* https://medium.com/@mbednarski/understanding-indexing-with-pytorch-gather-33717a84ebc4
        out = a.gather(-1, t) #* indexing the list with t
        # out = a[t] #* indexing the list with t
        #* reshape the output to match the shape of the input.
        #* [bathc_size, ] to [batch_size, 1, 1, 1]
        out = out.reshape(batch_size, *((1,) * (len(x_shape) - 1))).to(t.device)
        #* reshae (batch_size, *((1,)*3) 
        #* equivalent to
        #out = out.reshape(batch_size, *(1,1,1)) 
        return out
    
    
    def forward(self, x0, t, noise=None):
        #* generate noise if not given
        if noise is None:
            noise = torch.randn(x0.shape).to(self.device)
            
        #n_batch, n_ch, n_row, n_col = x0.shape
        #a_bar = self.alpha_bars[t]
        # noisy = img*sqrt(a_bar) + sqrt(1-a_bar)*noise
        #noisy = torch.sqrt(a_bar).reshape(n_batch, 1, 1, 1) * x0 + torch.sqrt(1 - a_bar).reshape(n_batch, 1, 1, 1) * noise

        ################ Forward process -- HuggingFace################
        ################# Q sample #################        
        #* Q(X_t | X_0) = sqrt(alpha_t) * X_0 + sqrt(1 - alpha) * N(0,I)  
        sqrt_alphas_cumprod_t = self.extract(self.sqrt_alphas_cumprod, t, x0.shape)
        sqrt_one_minus_alphas_cumprod_t = self.extract(self.sqrt_one_minus_alphas_cumprod, t, x0.shape)
        x_t = sqrt_alphas_cumprod_t * x0 + sqrt_one_minus_alphas_cumprod_t * noise
        return x_t
        #return noisy
        
        ################# P sample #################
    @torch.no_grad()
    def p_sample(self, x, t, noise=None, t_index=False):
        betas_t = self.extract(self.betas, t, x.shape)
        sqrt_one_minus_alphas_cumprod_t = self.extract(self.sqrt_one_minus_alphas_cumprod, t, x.shape)
        sqrt_recip_alphas_t = self.extract(self.sqrt_recip_alphas, t, x.shape)
        
        #model_mean  = sqrt_recip_alphas_t 
        #model_mean  = sqrt_recip_alphas_t *( x - betas_t *  self.forward(x, t)/ sqrt_one_minus_alphas_cumprod_t)
        model_mean  = sqrt_recip_alphas_t *( x - betas_t *  self.backward(x, t)/ sqrt_one_minus_alphas_cumprod_t)
        if t_index == 0:
            return model_mean
        else:
            posterior_variance_t = self.extract(self.posterior_variance, t, x.shape)
            noise = torch.randn_like(x).to(self.device)
            return model_mean + noise * torch.sqrt(posterior_variance_t)
    
    
    @torch.no_grad()
    def p_sample_loop(self, x_shape):
        device = self.device
        b = x_shape[0]
        
        #* initialize the noise
        img = torch.randn(x_shape).to(device)
        imgs = []
        
        
        for i in tqdm(range(0, self.n_steps)[::-1], desc="Sampling", total=self.n_steps):
            img = self.p_sample(img, torch.full((b,), i, dtype=torch.long).to(device), t_index=i)
            img = img.permute(0,2,3,1)
            # img = transforms.Lambda(lambda x: x.clamp_(0, 1))(img)
            imgs.append(img.cpu().numpy())
            
        return imgs
    #* for the U-net
    
    @torch.no_grad()
    def generate(self, x_shape):
        x = torch.randn(x_shape).to(self.device)
        for idx, t in enumerate(list(range(self.n_steps))[::-1]):
            time_tensor = (torch.ones(x_shape[0], 1) * t).to(self.device).long()
            
            #* back propogated noise
            eta_theta = self.backward(x, time_tensor)
            
            alpha_t  = self.alphas[t]
            alpha_bar_t = self.alphas_cumprod[t]
            
    
            x = (1 / torch.sqrt(alpha_t)) * (x - (1 - alpha_t) / (torch.sqrt(1 - alpha_bar_t)) * eta_theta)
            if t > 0:
                z = torch.randn(x_shape).to(self.device)
                beta_t = self.betas[t]
                sigma_t = torch.sqrt(beta_t)
                x = x + sigma_t * z
                
        return x.permute(0,2,3,1).cpu().numpy()
    
    def backward(self, x, t):
        return self.network(x, t)