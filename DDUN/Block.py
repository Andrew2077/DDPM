from torch import nn
import torch


class Block(nn.Module):
    def __init__(self, in_ch, out_ch, time_emb_dim, up=False, scale_img=False):
        super().__init__()
        self.time_mlp = nn.Linear(time_emb_dim, out_ch)  # * time embedding

        # * for upsampling blocks
        if up:
            self.conv1 = nn.Conv2d(2 * in_ch, out_ch, kernel_size=3, padding=1)
            if scale_img:
                self.transform = nn.Sequential(
                    nn.ConvTranspose2d(
                        out_ch, out_ch, kernel_size=4, stride=2, padding=1
                    ),
                )
            else:
                self.transform = nn.Sequential(
                    nn.ConvTranspose2d(
                        out_ch, out_ch, kernel_size=4, stride=2, padding=1
                    ),
                    nn.MaxPool2d(2, stride=2),
                )

        # * for downsampling blocks
        else:
            self.conv1 = nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1)
            self.transform = nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1)

        # * defining rest of the layers
        self.conv2 = nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1)
        # self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1)
        self.bnorm1 = nn.BatchNorm2d(out_ch)
        self.bnorm2 = nn.BatchNorm2d(out_ch)
        self.relu = nn.ReLU()

        self.down_scale = nn.MaxPool2d(2, stride=2)

    def forward(self, x, t):
        # * first conv
        h = self.relu(self.conv1(x))
        h = self.bnorm1(h)

        # #* time embedding
        time_emb = self.time_mlp(t)
        time_emb = self.relu(time_emb)

        # #* You may have to debug this part and permute the time embedding according to the shape of your input
        shape_premutation = 4- len(time_emb.shape)
        
        if shape_premutation == 2:
            time_emb = time_emb[(...,) + (None,) * shape_premutation]
        elif shape_premutation == 1:
            time_emb = torch.permute(time_emb[(...,) + (None,) * shape_premutation], (0, 2, 1, 3))  # * (4,1,64) -> (4,1,64,1) -> (4,64,1,1)

        # #* add time embedding to the output of the first conv
        h = h + time_emb

        # #* second conv

        h = self.bnorm2(self.relu(self.conv2(h)))
        h = self.transform(h)
        return h
