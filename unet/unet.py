import torch
import torch.nn as nn


class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels, dropout=0.0):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Dropout2d(dropout) if dropout > 0 else nn.Identity(),

            nn.Conv2d(out_channels, out_channels, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Dropout2d(dropout) if dropout > 0 else nn.Identity(),
        )

    def forward(self, x):
        return self.net(x)


class ResidualUNet(nn.Module):
    def __init__(self, in_channels=1, base_channels=8, dropout=0.1,
                 logspace=False, normalize=True):
        super().__init__()
        self.logspace = logspace
        self.normalize = normalize

        c1 = base_channels
        c2 = base_channels * 2
        c3 = base_channels * 4
        c4 = base_channels * 8
        c5 = base_channels * 16

        self.pool = nn.MaxPool2d(2)

        # Encoder
        self.down1 = DoubleConv(in_channels, c1, dropout=0.0)
        self.down2 = DoubleConv(c1, c2, dropout=dropout * 0.5)
        self.down3 = DoubleConv(c2, c3, dropout=dropout)
        self.down4 = DoubleConv(c3, c4, dropout=dropout)

        # Bottleneck (strongest dropout)
        self.bottleneck = DoubleConv(c4, c5, dropout=dropout)

        # Decoder
        self.up4 = nn.ConvTranspose2d(c5, c4, 2, stride=2)
        self.conv4 = DoubleConv(c4 * 2, c4, dropout=dropout)

        self.up3 = nn.ConvTranspose2d(c4, c3, 2, stride=2)
        self.conv3 = DoubleConv(c3 * 2, c3, dropout=dropout)

        self.up2 = nn.ConvTranspose2d(c3, c2, 2, stride=2)
        self.conv2 = DoubleConv(c2 * 2, c2, dropout=dropout * 0.5)

        self.up1 = nn.ConvTranspose2d(c2, c1, 2, stride=2)
        self.conv1 = DoubleConv(c1 * 2, c1, dropout=0.0)

        self.final = nn.Conv2d(c1, in_channels, kernel_size=1)

    def forward(self, x):
        input_img = x

        if self.logspace:
            x = torch.log1p(x)
            
        if self.normalize:
            # Compute per-image mean/std (over C, H, W)
            mean = x.mean(dim=(1, 2, 3), keepdim=True)
            std = x.std(dim=(1, 2, 3), keepdim=True)

            # Normalize
            x = (x - mean) / (std + 1e-6)

        # Encoder
        d1 = self.down1(x)
        d2 = self.down2(self.pool(d1))
        d3 = self.down3(self.pool(d2))
        d4 = self.down4(self.pool(d3))

        # Bottleneck
        b = self.bottleneck(self.pool(d4))

        # Decoder
        u4 = self.up4(b)
        u4 = torch.cat([u4, d4], dim=1)
        u4 = self.conv4(u4)

        u3 = self.up3(u4)
        u3 = torch.cat([u3, d3], dim=1)
        u3 = self.conv3(u3)

        u2 = self.up2(u3)
        u2 = torch.cat([u2, d2], dim=1)
        u2 = self.conv2(u2)

        u1 = self.up1(u2)
        u1 = torch.cat([u1, d1], dim=1)
        u1 = self.conv1(u1)

        background = self.final(u1)

        if self.normalize:
            # Undo normalization
            background = background * std + mean

        if self.logspace:
            background = torch.expm1(background)

        # Residual output
        clean = input_img - background

        return clean, background