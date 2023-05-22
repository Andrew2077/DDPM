from torch import nn


class Block(nn.Module):
    def __init__(
        self,
        shape,
        in_c,
        out_c,
        kernel_size=3,
        stride=1,
        padding=1,
        activation=None,
        normaliza=True,
    ):
        super().__init__()

        self.layer_norm = nn.LayerNorm(shape)  # * normalizing over the channels
        self.conv1 = nn.Conv2d(in_c, out_c, kernel_size, stride, padding)
        self.conv2 = nn.Conv2d(out_c, out_c, kernel_size, stride, padding)
        # * sigmoid linear unit SiLU : x * sigmoid(x)
        self.activation = nn.SiLU() if activation is None else activation
        self.norm = normaliza

    def forward(self, x):
        # * normalizing over the channels if needed
        out = self.layer_norm(x) if self.norm else x
        out = self.conv1(out)
        out = self.activation(out)
        out = self.conv2(out)
        out = self.activation(out)
        return out