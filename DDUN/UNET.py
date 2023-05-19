from torch import nn
import torch
from DDUN.Block import Block
from DDUN.Embedding import SinsoidalPositionalEmbeddings

class Diffusion_Unet(nn.Module):
    def __init__(self):
        super().__init__()

        image_channels = 1  # * grayscale image
        down_block_channels = (32, 64, 128, 256)
        up_channels = (256, 128, 64, 32)
        self.time_emb_dim = 32
        out_dim = 1

        # * Time Embedding
        self.time_mlp = nn.Sequential(
            SinsoidalPositionalEmbeddings(self.time_emb_dim),  # * positional embeddings
            nn.Linear(self.time_emb_dim, self.time_emb_dim),  # * linear layer
            nn.ReLU(),  # * activation
        )

        # * Initial Convolution
        self.conv0 = nn.Conv2d(
            image_channels, down_block_channels[0], kernel_size=3, padding=1
        )

        # * Downsampling Blocks
        self.downs = nn.ModuleList(
            [
                Block(
                    in_ch=down_block_channels[i],
                    out_ch=down_block_channels[i + 1],
                    time_emb_dim=self.time_emb_dim,
                )
                for i in range(len(down_block_channels) - 1)
            ]
        )

        # * Upsampling Blocks
        self.ups = nn.ModuleList(
            [
                Block(
                    in_ch=up_channels[i],
                    out_ch=up_channels[i + 1],
                    time_emb_dim=self.time_emb_dim,
                    up=True,
                    scale_img=False,
                )
                for i in range(len(up_channels) - 1)
            ]
        )

        # * Output Convolution
        self.outout = nn.Conv2d(up_channels[-1], out_dim, kernel_size=3, padding=1)

    # * forward pass
    def forward(self, x, t):
        # * Ebmedding time
        t = self.time_mlp(t)

        # * Initial Convolution
        x = self.conv0(x)

        # * Unet
        residual_input = []
        counter = 0
        # * append the input of each down block to the list
        for down in self.downs:
            x = down(x, t)
            residual_input.append(x)
            counter += 1

        # * pop the last element from the list
        for up in self.ups:
            x = torch.cat((x, residual_input.pop()), dim=1)
            x = up(x, t)

        # * output
        x = self.outout(x)
        return x
