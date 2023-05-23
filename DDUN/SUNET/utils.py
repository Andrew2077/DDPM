

import einops
import imageio
import matplotlib.pyplot as plt
import numpy as np
import torch


def show_images(images, title=""):
    # *detaching tensors to save ram
    if type(images) is torch.Tensor:
        images = images.detach().cpu().numpy()

    # * containing box
    fig = plt.figure(figsize=(8, 8))
    rows = int(len(images) ** 0.5)
    cols = round(len(images) / rows)
    # print(type(images), len(images))

    # * adjusting subplots
    idx = 0
    for r in range(rows):
        for c in range(cols):
            fig.add_subplot(rows, cols, idx + 1)
            
            if idx < len(images):
                #print(images[idx].shape)
                if images[idx].shape[0] == 1:
                    plt.imshow(images[idx][0], cmap="gray")
                else:
                    plt.imshow(images[idx][0].permute(1, 2, 0))
            plt.axis("off")
            idx += 1
    fig.suptitle(title, fontsize=20)
    fig.tight_layout()
    plt.show()
    
#* show the first batch of images
def show_first_batch(loader):
    for batch in loader :
        show_images(batch[0], title="First batch of images")
        break
    

from typing import List, Optional, Tuple


def show_forward(
    ddpm,
    loader,
    device,
    show_original=True,
    noise_percent: Optional[List[float]] = [0.25, 0.5, 0.75, 1],
):
    # Showing the forward process
    for batch in loader:
        imgs = batch[0]
        if noise_percent is None:
            noise_percent = [0.25, 0.5, 0.75, 1]
        if show_original:
            show_images(imgs, "Original images")

        for percent in (noise_percent if type(noise_percent) is list else [noise_percent]):     
            try:
                images_ = ddpm(imgs.to(device), int(percent * ddpm.n_steps) - 1)
            except:
                print("Wrong noise_percent input")
                print("Percentage has to be a list of floats from 0 to 1 or a float from 0 to 1")
                print('if you are not sure what to do, just use the default value of [0.25, 0.5, 0.75, 1] or pass None')
                break
            show_images(
                # images= ddpm(imgs.to(device),[int(percent * ddpm.n_steps) - 1 for _ in range(len(imgs))]),
                images=images_,
                title=f"DDPM Noisy images {(percent * 100)}%", 
            )
        break



def generate_new_images(
    ddpm,
    n_sample=16,
    device=None,
    frames_per_gif =100,
    gif_name="ddpm_samping.gif",
    n_c=1,
    img_h=28,
    img_w=28,
):
    #* line space of the steps 
    frames_idx = np.linspace(0, ddpm.n_steps, frames_per_gif ).astype(np.uint8)
    frames = []
    
    with torch.no_grad():
        if device is  None:
            device = ddpm.device
            
        #* generate noise
        x = torch.randn(n_sample, n_c, img_h, img_w).to(device) #* input shape
        
        #* generate images
        for idx, t in enumerate(list(range(ddpm.n_steps))[::-1]): #* reverse order
            #* estimating noise to be removed 
            
            time_tensor = (torch.ones(n_sample, 1) * t).to(device).long() #shape (batch_size, 1)
            
            #* back propogated noise
            eta_theta = ddpm.backward(x, time_tensor) #* shape (batch_size, n_c, img_h, img_w)
            
            alpha_t = ddpm.alphas[t]
            alpha_bar_t = ddpm.alphas_cumprod[t]
            
            x = (1/ torch.sqrt(alpha_t)) * (x - (1 -  alpha_t)/ (torch.sqrt(1-alpha_bar_t)) * eta_theta) 
            
            if t >0 :
                z = torch.randn(n_sample, n_c, img_h, img_w).to(device)
                beta_t = ddpm.betas[t]
                sigma_t = torch.sqrt(beta_t)
                
                #* applying Langevin Dynamics noise
                x = x + sigma_t * z 
                
                
            ## Creating the gif
            #* adding frames to the gif
            if idx in frames_idx or t == 0:
                normalized = x.clone()
                
                for i in range(len(normalized)):
                    normalized[i] -= torch.min(normalized[i])
                    normalized[i] *= 255 / torch.max(normalized[i])
                    
                #* resahimg to a square image
                frame = einops.rearrange(normalized, "(b1 b2) c h w -> (b1 h) (b2 w) c", b1=int(n_sample ** 0.5))
                frames.append(frame.cpu().numpy().astype(np.uint8))
        
    for i in range(int(len(frames)/5)):
        frames.append(frames[-1])
    #* saving the gif
    imageio.mimsave(gif_name, frames, format = 'GIF-PIL', fps = 90) #type: ignore
    
    # return frames
    #print(type(frames))
    return x


