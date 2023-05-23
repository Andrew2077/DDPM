import logging
import os
import torch
import torch.nn as nn
from matplotlib import pyplot as plt
from torch import optim
from torch.utils.tensorboard import SummaryWriter
#from tqdm.notebook import tqdm
from tqdm import tqdm

logging.basicConfig(
    format="%(asctime)s -%(levelname)s: %(message)s",
    level=logging.INFO,
    datefmt="%H:%M:%S",
)
from DDUN.ATTUNET.utils import *
from DDUN.ATTUNET.attunet import *


class Diffusion:
    def __init__(
        self,
        noise_steps : int=1000,
        beta_start=1e-4,
        beta_end=2e-2,
        img_size=28,
        n_ch=1,
        device="cuda",
    ):
        self.noise_steps = noise_steps
        self.beta_start_ = beta_start
        self.beta_end = beta_end
        self.img_size = img_size
        self.n_ch = n_ch
        self.device = device

        # * linear shedule
        self.beta = torch.linspace(
            self.beta_start_, self.beta_end, self.noise_steps
        ).to(self.device)
        self.alpha = 1 - self.beta
        self.alpha_hat = torch.cumprod(self.alpha, dim=0).to(self.device)

    def noise_images(self, x, t):
        sqrt_alpha_hat = torch.sqrt(self.alpha_hat[t])[
            :, None, None, None
        ]  # * (batch_size, 1, 1, 1)
        
        sqrt_one_minus_alpha_hat = torch.sqrt(1 - self.alpha_hat[t])[
            :, None, None, None
        ]  # * (batch_size, 1, 1, 1)
        epsilon = torch.randn_like(x).to(self.device)
        x_t = sqrt_alpha_hat * x + sqrt_one_minus_alpha_hat * epsilon
        return x_t, epsilon

    def sample_timesteps(self, n) -> torch.Tensor:
        return torch.randint(0, self.noise_steps, (n,)).to(self.device)

    def sample(self, model, n):
        logging.info("Sampling from the model")
        model.eval()
        with torch.no_grad():
            x = torch.randn(n, self.n_ch, self.img_size, self.img_size).to(self.device)
            for i in tqdm(reversed(range(1, self.noise_steps)), position=0):
                t = (torch.ones(n) * i).long().to(self.device)
                predicted_noise = model(x, t)
                alpha = self.alpha[i][:, None, None, None]
                alpha_hat = self.alpha_hat[i][:, None, None, None]
                beta = self.beta[i][:, None, None, None]

                if i > 1:
                    noise = torch.randn_like(x)  # .to(self.device)
                else:
                    noise = torch.randn_like(x)  # .to(self.device)
                    
                #* sampling equation P(x_1})
                x = (
                    1
                    / torch.sqrt(alpha)
                    * (x - ((1 - alpha) / torch.sqrt(1 - alpha_hat)) * predicted_noise)
                    + torch.sqrt(beta) * noise
                )

        model.train()
        x = (x.clamp(-1, 1) + 1) / 2
        
        x = (x * 255).type(torch.uint8)
        return x


# * training loop function
def train(args):
    setup_logging(args.run_name)
    device = args.device
    dataloader = load_data(args)
    model = AttUnet().to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    mse = nn.MSELoss()
    diffusion= Diffusion(img_size= args.images_size, device= device)
    logger = SummaryWriter(os.path.join('results', args.run_name))
    num_batches = len(dataloader)
    
    for epoch in range(args.epochs):
        logging.info(f"Starting epoch {epoch}:")
        pbar = tqdm(dataloader)
        print(len(dataloader))
        for i, (images, _) in enumerate(dataloader):
            images = images.to(device)
            t = diffusion.sample_timesteps(images.shape[0]).to(device)
            #t = torch.randint(0, diffusion.noise_steps, (images.shape[0],)).to(device)
            
            x_t, real_noise = diffusion.noise_images(images, t)
            noise = model(images, x_t)
            
            loss = mse(noise, real_noise)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            pbar.set_postfix(MSE = loss.item())
            logger.add_scalar("MSE", loss.item(), global_step=epoch*num_batches+i)
            
            
        sampled_images = diffusion.sample(model, n = images.shape[0])
        save_images(sampled_images, os.path.join('results', args.run_name, f"{epoch}.png"))
        torch.save(model.state_dict(), os.path.join('models', args.run_name, f"{epoch}.pth"))
        
def launch():
    import argparse
    parser = argparse.ArgumentParser()
    args = parser.parse_args()
    args.run_name = "attunet_DDPM"
    args.epochs = 10
    args.batch_size = 16
    args.images_size = 28
    args.dataset_path = r"data"
    args.device = "cuda"
    args.in_ch = 1
    args.lr = 3e-4
    train(args)
    
if __name__ == "__main__":
    launch()