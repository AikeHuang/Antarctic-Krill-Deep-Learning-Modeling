import torch
import torch.nn as nn
import torch.nn.functional as F

class ConvBlock(nn.Module):
    def __init__(self, in_c, out_c, dropout_p=0.1):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_c, out_c, 3, padding=1),
            nn.BatchNorm2d(out_c),
            nn.LeakyReLU(),
            nn.Dropout2d(p=dropout_p),

            nn.Conv2d(out_c, out_c, 3, padding=1),
            nn.BatchNorm2d(out_c),
            nn.LeakyReLU(),
            nn.Dropout2d(p=dropout_p)
        )

    def forward(self, x):
        return self.block(x)


class DownSample(nn.Module):
    def __init__(self, ch):
        super().__init__()
        self.layer = nn.Sequential(
            nn.Conv2d(ch, ch, 3, stride=2, padding=1),
            nn.BatchNorm2d(ch),
            nn.LeakyReLU()
        )

    def forward(self, x):
        return self.layer(x)


class UpSample(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv1x1 = nn.Conv2d(in_ch, out_ch, 1)

    def forward(self, x, skip):
        x = F.interpolate(x, scale_factor=2, mode='nearest')
        x = self.conv1x1(x)
        x = torch.cat([x, skip], dim=1)
        return x


class UNet(nn.Module):
    def __init__(self, in_channels=6, num_classes=1, dropout_p=0.1):
        super().__init__()
        
        self.c1 = ConvBlock(in_channels, 32, dropout_p=dropout_p)
        self.d1 = DownSample(32)
        self.c2 = ConvBlock(32, 64, dropout_p=dropout_p)
        self.d2 = DownSample(64)
        self.c3 = ConvBlock(64, 128, dropout_p=dropout_p)
        self.d3 = DownSample(128)
        self.c4 = ConvBlock(128, 256, dropout_p=dropout_p)

        self.u1 = UpSample(256, 128)
        self.c5 = ConvBlock(256, 128, dropout_p=dropout_p)
        self.u2 = UpSample(128, 64)
        self.c6 = ConvBlock(128, 64, dropout_p=dropout_p)
        self.u3 = UpSample(64, 32)
        self.c7 = ConvBlock(64, 32, dropout_p=dropout_p)

        self.out = nn.Conv2d(32, num_classes, 1)

    def forward(self, x):
        r1 = self.c1(x)
        r2 = self.c2(self.d1(r1))
        r3 = self.c3(self.d2(r2))
        r4 = self.c4(self.d3(r3))

        x = self.u1(r4, r3)
        x = self.c5(x)
        x = self.u2(x, r2)
        x = self.c6(x)
        x = self.u3(x, r1)
        x = self.c7(x)

        return self.out(x)