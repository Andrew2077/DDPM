import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from tqdm.notebook import tqdm
from matplotlib import pyplot as plt
from PIL import Image
import imageio
import einops
import numpy as np


class MarkovDDPM:
    def __init__(
        self,
        noise_steps,
        start=1e-4,
        end=2e-2,
        image_ch=1,
        image_size=28,
        num_classes = None,
        device=None,
    ):
        ################ Define the process -- Outlier################
        self.noise_steps = noise_steps
        self.start = start
        self.end = end
        self.image_ch = image_ch
        self.image_size = image_size
        self.device = device

        self.beta = torch.linspace(start, end, noise_steps).to(device)
        self.alpha = 1 - self.beta
        self.alpha_hat = torch.cumprod(self.alpha, dim=0).to(device)
        ################ Define the process -- HuggingFace################
        # * Alphas
        # self.betas = torch.linspace(start, end, n_steps).to(device)
        # self.alphas = 1 - self.betas
        # self.alphas_cumprod = torch.cumprod(self.alphas, dim=0).to(device)
        # self.alphas_cumprod_prev = F.pad(self.alphas_cumprod[:-1], (1, 0), value=1.0)
        # self.sqrt_recip_alphas = torch.sqrt(1.0 / self.alphas)

        # #* calculation for diffusion q(x_t| x_{t-1})
        # # self.alpha_bars = torch.cumprod(self.alphas, dim=0).to(device)
        # self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        # self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - self.alphas_cumprod)

        # #* calculation for posterior  q(x_{t-1}| x_t, x_0)})
        # self.posterior_variance = self.betas * (1. - self.alphas_cumprod_prev) / (1. - self.alphas_cumprod)

    def noise_images(self, x0, t):
        sqrt_alpha_hat = torch.sqrt(self.alpha_hat[t])[:, None, None, None]
        sqrt_one_minus_alpha_hat = torch.sqrt(1 - self.alpha_hat[t])[
            :, None, None, None
        ]
        epsilon = torch.randn_like(x0).to(self.device)
        x_t = sqrt_alpha_hat * x0 + sqrt_one_minus_alpha_hat * epsilon
        return x_t, epsilon

    # @torch.no_grad()
    def generate(self, model, x_shape, labels = None, save_gen_hist=False,  cfg_scale=3):
        model.eval()
        if save_gen_hist:
            gen_hist = []
        with torch.no_grad():
            x = torch.rand(x_shape).to(self.device)
            # print(self.noise_steps)
            # for idx, i in enumerate(tqdm(range(self.noise_steps)[::-1])):
            for idx, i in enumerate(range(1, self.noise_steps)[::-1]):
                # * create a time tensor for time t, with shape (batch_size, 1)
                # t = (torch.ones(torch.tensor(x_shape[0])) * i).long().to(self.device)
                t = (torch.ones(x_shape[0], 1) * i).to(self.device).long()
                
                
                # * predicted noise

                predicted_noise = model(x, t, labels)
                if cfg_scale > 0:
                    uncodtional_prediction = model(x, t, labels=None)
                    predicted_noise = torch.lerp(predicted_noise, uncodtional_prediction, cfg_scale)

                
                # * retrieve alpha, alpha_hat, beta for time t
                # * reshape them to (batch_size, 1, 1, 1)
                alpha = self.alpha[i].repeat(x_shape[0], 1, 1, 1)
                alpha_hat = self.alpha_hat[i].repeat(x_shape[0], 1, 1, 1)
                beta = self.beta[i].repeat(x_shape[0], 1, 1, 1)
                if i > 1:
                    # * if t > 1, sample noise from N(0,1)
                    noise = torch.randn_like(x).to(self.device)
                else:
                    # * if t = 1, noise will be zeros
                    noise = torch.zeros_like(x).to(self.device)

                # * sample from the diffusion process
                x = (
                    1
                    / torch.sqrt(alpha)
                    * (x - (1 - alpha) / torch.sqrt(1 - alpha_hat) * predicted_noise)
                    + torch.sqrt(beta) * noise
                )
                if save_gen_hist:
                    gen_hist.append(x)
        model.train()
        # x = (x.clamp(-1, 1) + 1) / 2
        # x = (x * 255).type(torch.uint8)
        # x= x.permute(0, 2, 3, 1)
        if save_gen_hist:
            return x, gen_hist
        return x

    def generate_new_images(self, model, batch_size):
        shape = (batch_size, self.image_ch, self.image_size, self.image_size)
        generated_batch = self.generate(model, shape)

        generated_batch = generated_batch.to("cpu")
        fig = plt.figure(figsize=(8, 8))
        rows = int(len(generated_batch) ** 0.5)
        cols = round(len(generated_batch) / rows)
        idx = 0
        for r in range(rows):
            for c in range(cols):
                fig.add_subplot(rows, cols, idx + 1)
                img = generated_batch[idx].permute(1, 2, 0)
                img = img.to("cpu").detach().numpy()

                plt.imshow(img, cmap="gray")
                plt.axis("off")
                idx += 1

        fig.tight_layout()
        plt.show()

    def save_gen_into_gif(self, gen_hist, gif_name=None):
        frames = []
        for idx, tensor in enumerate(gen_hist[-2 * int(len(gen_hist) / 3) :]):
            if idx % 9 == 0:
                normalized = tensor.clone()

                for i in range(len(normalized)):
                    normalized[i] -= torch.min(normalized[i])
                    normalized[i] *= 255 / torch.max(normalized[i])

                # * resahimg to a square image
                frame = einops.rearrange(
                    normalized,
                    "(b1 b2) c h w -> (b1 h) (b2 w) c",
                    b1=int(tensor.shape[0] ** 0.5),
                )
                frame = frame.cpu().numpy().astype(np.uint8)
                frame = np.squeeze(frame, axis=2)
                # * converting to PIL image
                frame = Image.fromarray(frame)
                frame = frame.resize((1024, 1024))
                frame = np.array(frame)
                frames.append(frame)
        for i in range(18):
            frames.append(frames[-1])
        if gif_name == None:
            gif_name = "SDDPM_results"
        imageio.mimsave(f"{gif_name}.gif", frames, format="GIF-PIL", fps=100000)  # type: ignore
        print(f"gif with {len(frames)} frames saved")
        plt.imshow(frames[-1], cmap="gray")
        plt.axis("off")
