import torch
import torch.nn as nn
import torch.nn.functional as F
from DDUN.ATTUNET.embedding import pos_encoding


##* Double Convolutional Block
class DoubleConv(nn.Module):
    def __init__(self, in_ch, out_ch, mid_ch=None, residual=False):
        super().__init__()
        self.in_ch = in_ch
        self.out_ch = out_ch
        self.mid_ch = mid_ch
        self.residual = residual
        if not mid_ch:
            mid_ch = out_ch
            
        
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_ch, mid_ch, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(1, mid_ch),  # * read about it
            nn.GELU(),  # * read about it
            nn.Conv2d(mid_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(1, out_ch),  # * read about it
        )
        
                # self.double_conv = nn.Sequential(
        
        self.l1 = nn.Conv2d(in_ch, mid_ch, kernel_size=3, padding=1, bias=False)
        self.l2 = nn.GroupNorm(1, mid_ch),  # * read about it
        self.l3 = nn.GELU()  # * read about it
        self.l4 = nn.Conv2d(mid_ch, out_ch, kernel_size=3, padding=1, bias=False)
        self.l5 = nn.GroupNorm(1, out_ch),  # * read about it
        

    def forward(self, x):
        if self.residual:
            # print("x", x.shape)
            # print("in_ch", self.in_ch, "out_ch", self.out_ch, "mid_ch", self.mid_ch)
            # res_x = self.l1(x)
            # print(type(res_x))
            # print("l1", res_x.shape)
            # res_x = self.l2(res_x)
            # res_x = self.l3(res_x)
            # res_x = self.l4(res_x)
            # res_x = self.l5(res_x)
           
            res_x = self.double_conv(x)
            print("x", x.shape , "res_x", res_x.shape)
            conneciion = x + res_x
            return F.gelu(conneciion)  # * create a residual connection
        else:
            return self.double_conv(x)


class Down(nn.Module):
    def __init__(self, in_ch, out_ch, embed_dim=256):
        super().__init__()

        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_ch, out_ch, embed_dim, residual=True),
            DoubleConv(out_ch, out_ch, embed_dim, residual=False),
        )

        self.embed_layer = nn.Sequential(
            nn.SiLU(),
            nn.Linear(embed_dim, embed_dim),
        )

    def forward(self, x, t):
        x = self.maxpool_conv(x)
        t = self.embed_layer(t)
        return x + t


class Up(nn.Module):
    def __init__(self, in_ch, out_ch, embed_dim=256):
        super().__init__()

        self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
        self.conv = nn.Sequential(
            DoubleConv(in_ch, out_ch, embed_dim, residual=True),
            DoubleConv(out_ch, out_ch, embed_dim, in_ch // 2),  # * add mid_ch
        )

        self.embed_layer = nn.Sequential(
            nn.SiLU(),
            nn.Linear(embed_dim, embed_dim),
        )

    def forward(self, x, skip_x, t):
        x = self.up(x)
        x = torch.cat([x, skip_x], dim=1)

        x = self.conv(x)
        emb = self.embed_layer(t)

        return x + emb


class SelfAttention(nn.Module):
    def __init__(self, channels, size):
        super(SelfAttention, self).__init__()

        self.channels = channels
        self.size = size

        self.MultiHeadAtt = nn.MultiheadAttention(
            embed_dim=channels, num_heads=1, batch_first=True
        )
        self.lnorm = nn.LayerNorm([channels])
        self.ff_self = nn.Sequential(
            nn.LayerNorm([channels]),
            nn.Linear(channels, channels),
            nn.GELU(),
            nn.Linear(channels, channels),
        )

    def forward(self, x):
        x = x.view(-1, self.channels, self.size * self.size).swapaxes(1, 2)
        x_ln = self.lnorm(x)
        attention_value, _ = self.MultiHeadAtt(x_ln, x_ln, x_ln)
        attention_value = attention_value + x
        attention_value = self.ff_self + attention_value
        x = attention_value.swapaces(2, 1).view(-1, self.channels, self.size, self.size)
        return x


class AttUnet(nn.Module):
    def __init__(self, c_in=1, c_out=1, time_dim=256, image_size = 64, device="cuda", attention=True):
        super().__init__()
        self.add_attention = attention

        self.device = device
        self.time_dim = time_dim
        self.c_in = c_in
        self.c_out = c_out

        # * downsample
        self.inc = DoubleConv(c_in, image_size)
        self.down1 = Down(image_size, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 256)

        if attention:
            self.sa1 = SelfAttention(128, 32)
            self.sa2 = SelfAttention(256, 16)
            self.sa3 = SelfAttention(256, 8)
            self.sa4 = SelfAttention(128, 8)
            self.sa5 = SelfAttention(64, 32)
            self.sa6 = SelfAttention(64, 64)

        # * multi head attention bottleneck
        self.bot1 = DoubleConv(256, 512)
        self.bot2 = DoubleConv(512, 512)
        self.bot3 = DoubleConv(512, 256)

        # * upsample
        self.up1 = Up(512, 128)
        self.up2 = Up(256, 64)
        self.up3 = Up(128, 64)

        # * output
        self.output = nn.Conv2d(64, c_out, 1)



    def forward(self, x, t):
        t = t.unsqueeze(-1).type(torch.float)
        t = pos_encoding(t, self.time_dim)

        x1 = self.inc(x)
        # print(x1.shape)

        x2 = self.down1(x1, t)

        if self.add_attention:
            x2 = self.sa1(x2)

        x3 = self.down2(x2, t)
        if self.add_attention:
            x3 = self.sa2(x3)

        x4 = self.down3(x3, t)
        if self.add_attention:
            x4 = self.sa3(x4)

        x4 = self.bot1(x4)
        x4 = self.bot2(x4)
        x4 = self.bot3(x4)

        x = self.up1(x4, x3, t)
        if self.add_attention:
            x = self.sa4(x)

        x = self.up2(x, x2, t)
        if self.add_attention:
            x = self.sa5(x)

        x = self.up3(x, x1, t)
        if self.add_attention:
            x = self.sa6(x)

        out = self.output(x)
        return out
