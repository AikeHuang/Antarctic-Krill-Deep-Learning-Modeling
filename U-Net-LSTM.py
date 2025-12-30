import torch
import torch.nn as nn
import torch.nn.functional as F

class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, dropout_p=0.1):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(),
            nn.Dropout2d(p=dropout_p),
            
            nn.Conv2d(out_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(),
            nn.Dropout2d(p=dropout_p)
        )

    def forward(self, x):
        return self.block(x)


class DownSample(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.layer = nn.Sequential(
            nn.Conv2d(channels, channels, 3, stride=2, padding=1),
            nn.BatchNorm2d(channels),
            nn.LeakyReLU()
        )

    def forward(self, x):
        return self.layer(x)


class UpSample(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.conv1x1 = nn.Conv2d(in_channels, in_channels // 2, kernel_size=1)

    def forward(self, x, skip):
        x = F.interpolate(x, scale_factor=2, mode='nearest')
        x = self.conv1x1(x)
        return torch.cat([x, skip], dim=1)


class ConvLSTMCell(nn.Module):
    def __init__(self, in_channels, hidden_channels, kernel_size=3):
        super().__init__()
        padding = kernel_size // 2
        self.conv = nn.Conv2d(in_channels + hidden_channels,
                              4 * hidden_channels,
                              kernel_size,
                              padding=padding)

    def forward(self, x, h, c):
        combined = torch.cat([x, h], dim=1)
        gates = self.conv(combined)
        i, f, o, g = torch.chunk(gates, 4, dim=1)
        
        i = torch.sigmoid(i)
        f = torch.sigmoid(f)
        o = torch.sigmoid(o)
        g = torch.tanh(g)
        
        c_next = f * c + i * g
        h_next = o * torch.tanh(c_next)
        return h_next, c_next


class UNetLSTM(nn.Module):
    def __init__(self, in_channels=6, hidden_channels=256, dropout_p=0.1):
        super().__init__()
        self.c1 = ConvBlock(in_channels, 32, dropout_p=dropout_p)
        self.d1 = DownSample(32)
        self.c2 = ConvBlock(32, 64, dropout_p=dropout_p)
        self.d2 = DownSample(64)
        self.c3 = ConvBlock(64, 128, dropout_p=dropout_p)
        self.d3 = DownSample(128)
        
        self.convlstm = ConvLSTMCell(128, hidden_channels)
        
        self.u1 = UpSample(hidden_channels)
        self.c4 = ConvBlock(256, 128, dropout_p=dropout_p)
        self.u2 = UpSample(128)
        self.c5 = ConvBlock(128, 64, dropout_p=dropout_p)
        self.u3 = UpSample(64)
        self.c6 = ConvBlock(64, 32, dropout_p=dropout_p)
        self.out = nn.Conv2d(32, 1, kernel_size=1)

    def forward(self, x_seq):
        B, T, C, H, W = x_seq.shape
        h = torch.zeros(B, 256, H // 8, W // 8, device=x_seq.device)
        c = torch.zeros_like(h)
        
        for t in range(T):
            x = x_seq[:, t]
            r1 = self.c1(x)
            r2 = self.c2(self.d1(r1))
            r3 = self.c3(self.d2(r2))
            x_down = self.d3(r3)
            h, c = self.convlstm(x_down, h, c)
            
        x = self.c4(self.u1(h, r3))
        x = self.c5(self.u2(x, r2))
        x = self.c6(self.u3(x, r1))
        return self.out(x)