import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
import numpy as np
from pathlib import Path

class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels, dropout=0.0):
        super().__init__()
        
        # Branch 1: First conv layer group
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.InstanceNorm2d(out_channels, affine=True),
            nn.LeakyReLU(inplace=True),
            nn.Dropout2d(dropout) if dropout > 0 else nn.Identity(),
        )
        
        # Branch 2: Second conv layer group
        self.conv2 = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.InstanceNorm2d(out_channels, affine=True),
            nn.Dropout2d(dropout) if dropout > 0 else nn.Identity(),
        )
        
        # Final activation applied AFTER the residual addition
        self.act = nn.LeakyReLU(inplace=True)

        # The Skip Connection (Shortcut) path
        if in_channels != out_channels:
            self.shortcut = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, bias=False)
        else:
            self.shortcut = nn.Identity()

    def forward(self, x):
        # 1. Compute the residual path
        residual = self.shortcut(x)
        
        # 2. Compute the main convolutional path
        out = self.conv1(x)
        out = self.conv2(out) # Ends with dropout, no activation yet
        
        # 3. Add the shortcut
        out = out + residual
        
        # 4. Apply activation last
        return self.act(out)


class ResidualUNet(nn.Module):
    def __init__(self, in_channels=1, base_channels=8, dropout=0.1,
                 logspace=False, normalize=True, predict_background=True,
                 reduced_channels=False):
        super().__init__()
        self.in_channels = in_channels
        self.base_channels = base_channels
        self.dropout = dropout
        self.logspace = logspace
        self.normalize = normalize
        self.predict_background = predict_background
        self.reduced_channels = reduced_channels

        if reduced_channels:
            c1 = base_channels
            c2 = base_channels + 1
            c3 = base_channels + 2
            c4 = base_channels + 3
            c5 = base_channels + 4
        else:
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

        self.final = nn.Conv2d(c1 + in_channels, in_channels, kernel_size=1)

    def forward(self, x):
        if self.logspace:
            x = torch.log1p(x)

        input_img = x
            
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

        output = self.final(torch.cat([u1, x], dim=1))

        if self.normalize:
            # Undo normalization
            output = output * std + mean

        if self.predict_background:
            # Residual output
            if self.training:
                # background = torch.nn.functional.leaky_relu(output, negative_slope=0.01)
                # clean = torch.nn.functional.leaky_relu(input_img - background, negative_slope=0.01)
                background = torch.nn.functional.softplus(output)
                clean = torch.nn.functional.softplus(input_img - background)
            else:
                # background = torch.relu(output)
                # clean = torch.relu(input_img - background)
                background = torch.nn.functional.softplus(output)
                clean = torch.nn.functional.softplus(input_img - background)
        else:
            if self.training:
                clean = torch.nn.functional.leaky_relu(output, negative_slope=0.01)
            else:
                clean = torch.clamp_max_(output, input_img)
                clean = torch.clamp_min_(output, 0)

        if self.logspace and not self.training:
            clean = torch.expm1(clean)

        return clean
    
    def predict(self, x):
        if isinstance(x, np.ndarray):
            x = torch.from_numpy(x)

        if len(x.shape) < 2:
            raise ValueError(f"Invalid shape {x.shape}")
        elif len(x.shape) == 2:
            x = x[None, None]
        elif len(x.shape) == 3:
            x = x[:, None]

        if x.dtype != torch.float32:
            x = x.float()

        device = self.final.weight.device
        x = x.to(device)

        self.eval()
        with torch.no_grad():
            clean = self(x)
            return clean.squeeze().cpu().numpy()
        
    def batch_predict(self, x, batch_size=100):
        if isinstance(x, np.ndarray):
            x = torch.from_numpy(x)

        if len(x.shape) < 2:
            raise ValueError(f"Invalid shape {x.shape}")
        elif len(x.shape) == 2:
            x = x[None, None]
        elif len(x.shape) == 3:
            x = x[:, None]

        if x.dtype != torch.float32:
            x = x.float()

        dataset = TensorDataset(x)
        dataloader = DataLoader(dataset, batch_size=batch_size)

        device = self.final.weight.device
        outputs = []
        self.eval()
        for batch in dataloader:
            batch = batch[0].to(device)
            with torch.no_grad():
                output = self(batch)
            outputs.append(output)

        return torch.cat(outputs).cpu()
    
    def save(self, path, epoch=None, optimizer=None, avg_loss=None):
        torch.save({
            # Weights
            "model_state_dict": self.state_dict(),
            # Parameters
            "model_params":{
                "in_channels": self.in_channels,
                "base_channels": self.base_channels,
                "dropout": self.dropout,
                "logspace": self.logspace,
                "normalize": self.normalize,
                "predict_background": self.predict_background,
                "reduced_channels": self.reduced_channels,
            },
            # Training state
            "epoch": epoch,
            "optimizer_state_dict": optimizer.state_dict() if optimizer else None,
            "loss": avg_loss,
        }, path)

    @staticmethod
    def load(path, pattern=None):
        if pattern != None:
            path = next(Path(path).glob(pattern))
        c = torch.load(path)

        model = ResidualUNet(**c["model_params"])
        model.load_state_dict(c['model_state_dict'])

        return model, c
