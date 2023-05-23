import torch
from torch import nn
import torch.nn.functional as F
from DDUN.SUNET.embedding import SinsuoidalPostionalEmbedding
from DDUN.SUNET.block import Block


class Unet(nn.Module):
    def __init__(self, noise_steps=1024, time_emb_dim=256, device=None):
        super().__init__()
        self.device = device
        # * sinusoidal positional embedding
        self.time_embed = nn.Embedding(noise_steps, time_emb_dim, device=device)
        self.time_embed.weight.data = SinsuoidalPostionalEmbedding(time_emb_dim)(
            torch.arange(noise_steps, device=device), device
        )
        self.time_embed.requires_grad_(False)

        # * Brain_puler Embedding
        # self.time_embed = nn.Embedding(noise_steps, time_emb_dim)
        # self.time_embed.weight.data = sinusoidal_embedding(noise_steps, time_emb_dim)
        # self.time_embed.requires_grad_(False)

        ################### First Half of the U-net ###################

        ##*part 1
        self.te1 = self._make_te(time_emb_dim, 1)
        # * te1 is the time embedding for the first half of the U-net
        self.block1 = nn.Sequential(
            # *(batch_size, 1, 28, 28) to (batch_size, 10, 28, 28)
            Block(shape=(1, 28, 28), in_c=1, out_c=10),
            Block(shape=(10, 28, 28), in_c=10, out_c=10),
            Block(shape=(10, 28, 28), in_c=10, out_c=10),
        )
        self.down1 = nn.Conv2d(10, 10, 2, 2)
        # * (batch_size, 10, 28, 28) to (batch_size, 10, 14, 14)
        # * try MaxPool2d instead of Conv2d

        ##*part 2
        self.te2 = self._make_te(time_emb_dim, 10)
        # * te2 is the time embedding for the second half of the U-net
        self.block2 = nn.Sequential(
            # * (batch_size, 10, 14, 14) to (batch_size, 20, 14, 14)
            Block(shape=(10, 14, 14), in_c=10, out_c=20),
            Block(shape=(20, 14, 14), in_c=20, out_c=20),
            Block(shape=(20, 14, 14), in_c=20, out_c=20),
        )
        self.down2 = nn.Conv2d(20, 20, 2, 2)
        # * (batch_size, 20, 14, 14) to (batch_size, 20, 7, 7)

        ##* part 3
        self.te3 = self._make_te(time_emb_dim, 20)
        # * te3 is the time embedding for the third half of the U-net
        self.block3 = nn.Sequential(
            # * (batch_size, 20, 7, 7) to (batch_size, 40, 7, 7)
            Block(shape=(20, 7, 7), in_c=20, out_c=40),
            Block(shape=(40, 7, 7), in_c=40, out_c=40),
            Block(shape=(40, 7, 7), in_c=40, out_c=40),
        )

        # * down3 is the last down sampling layer
        # * (batch_size, 40, 7, 7) to (batch_size, 40, 3, 3)
        #! maxpooling won't work here, may be try
        self.down3 = nn.Sequential(
            nn.Conv2d(40, 40, 2, 2),
            nn.SiLU(),
            nn.Conv2d(40, 40, 1, 1),
        )

        ## Bottleneck
        # upsample the image from (batch_size, 40, 3, 3) to (batch_size, 20, 3, 3)

        # * time embedding for the bottleneck
        self.te_mid = self._make_te(time_emb_dim, 40)
        self.bottleneck = nn.Sequential(
            # * (batch_size, 40, 3, 3) to (batch_size, 40, 3, 3)
            Block(shape=(40, 3, 3), in_c=40, out_c=20),
            Block(shape=(20, 3, 3), in_c=20, out_c=20),
            Block(shape=(20, 3, 3), in_c=20, out_c=40),
        )

        ################### Second Half of the U-net ###################

        # * first upsample block
        self.up1 = nn.Sequential(
            # * (batch_size, 20, 3, 3) to (batch_size, 20, 7, 7)
            nn.ConvTranspose2d(
                40, 40, 4, 2, 1
            ),  # kernel_size =4 , stride = 2, padding = 1
            nn.SiLU(),
            nn.ConvTranspose2d(
                40, 40, 2, 1
            ),  # kernel_size = 2, stride = 1, padding = 0
        )

        # * 4th time embedding layer
        self.te4 = self._make_te(time_emb_dim, 80)

        # * feature extraction block, inserted after the first upsample block
        # * to extract features from the skip connection
        # * reduce the number of channels from 80 to 20
        self.block4 = nn.Sequential(
            # * (batch_size, 80, 7, 7) to (batch_size, 20, 7, 7)
            Block(shape=(80, 7, 7), in_c=80, out_c=40),
            Block(shape=(40, 7, 7), in_c=40, out_c=20),
            Block(shape=(20, 7, 7), in_c=20, out_c=20),
        )
        # * second upsample block
        self.up2 = nn.ConvTranspose2d(20, 20, 4, 2, 1)
        # * (batch_size, 20, 7, 7) to (batch_size, 20, 14, 14)

        # * 5th time embedding layer
        self.te5 = self._make_te(time_emb_dim, 40)
        self.b5 = nn.Sequential(
            # * (batch_size, 40, 14, 14) to (batch_size, 10, 14, 14)
            Block(shape=(40, 14, 14), in_c=40, out_c=20),
            Block(shape=(20, 14, 14), in_c=20, out_c=10),
            Block(shape=(10, 14, 14), in_c=10, out_c=10),
        )

        # * third upsample block
        self.up3 = nn.ConvTranspose2d(10, 10, 4, 2, 1)
        # * output time embedding layer
        self.te_out = self._make_te(time_emb_dim, 20)

        # * output block
        self.out_block = nn.Sequential(
            # * (batch_size, 10, 28, 28) to (batch_size, 10, 28, 28)
            Block(shape=(20, 28, 28), in_c=20, out_c=10),
            Block(shape=(10, 28, 28), in_c=10, out_c=10),
            Block(shape=(10, 28, 28), in_c=10, out_c=10, normalizer=False),
        )

        # * output convolution layer
        # * (batch_size, 10, 28, 28) to (batch_size, 1, 28, 28)
        self.conv_out = nn.Conv2d(10, 1, 3, 1, 1)

    def _make_te(self, dim_in, dim_out):
        return nn.Sequential(
            nn.Linear(dim_in, dim_out),
            nn.ReLU(),
            nn.Linear(dim_out, dim_out),
        ).to(self.device)

    def forward(self, x, t):
        
        t = t.long().to(self.device)
        x = x.to(self.device)
        t = self.time_embed(t)
        # t = t.to(x.device).long()
        n = len(x)  # * batch size

        # * x + te1(t) where te1 is the time embedding for the first half of the U-net
        # * apply dynamic additive noise to the input with the time embedding

        # * (batch_size, 1, 28, 28) to (batch_size, 10, 28, 28)
        # time_embedded = self.te1(t).reshape(n, -1, 1, 1)
        

        assert (
            len(self.te1(t).reshape(n, -1, 1, 1).shape) == len(x.shape)
            and self.te1(t).reshape(n, -1, 1, 1).shape[0] == x.shape[0]
        ), "shape of the time embedding and the input image should be the same"

        out1 = self.block1(x + self.te1(t).reshape(n, -1, 1, 1))

        # * (batch_size, 10, 28, 28) to (batch_size, 20, 14, 14)
        out2 = self.block2(self.down1(out1) + self.te2(t).reshape(n, -1, 1, 1))

        # * (batch_size, 20, 14, 14) to (batch_size, 40, 7, 7)
        out3 = self.block3(self.down2(out2) + self.te3(t).reshape(n, -1, 1, 1))

        # * (batch_size, 40, 7, 7) to (batch_size, 40, 3, 3)
        bottelneck = self.bottleneck(
            self.down3(out3) + self.te_mid(t).reshape(n, -1, 1, 1)
        )

        # * (batch_size, 40, 3, 3) to (batch_size, 20, 3, 3)
        skip1 = torch.cat(
            (out3, self.up1(bottelneck)), dim=1
        )  # * [batch_size, 80, 7, 7])

        # * [batch_size, 80, 7, 7]) to (batch_size, 20, 7, 7)
        out4 = self.block4(
            skip1 + self.te4(t).reshape(n, -1, 1, 1)
        )  # * (batch_size, 20, 7, 7)

        # * (batch_size, 20, 7, 7) to (batch_size, 20, 14, 14)
        skip2 = torch.cat((out2, self.up2(out4)), dim=1)  # * (batch_size, 40, 14, 14)

        # * (batch_size, 40, 14, 14) to (batch_size, 10, 14, 14)
        out5 = self.b5(skip2 + self.te5(t).reshape(n, -1, 1, 1))

        # *model output
        skip3 = torch.cat((out1, self.up3(out5)), dim=1)  # * (batch_size, 20, 28, 28)
        out6 = self.out_block(
            skip3 + self.te_out(t).reshape(n, -1, 1, 1)
        )  # * (batch_size, 10, 28, 28)

        #* predicted noise
        diffused_batch = self.conv_out(out6)  # * (batch_size, 1, 28, 28)
        return diffused_batch
