"""Illumination Enhancement Branch (Paper Sec. 3.2).

    L_enh = IllumUNet(L_low)                                   (Eq. 8)

The illumination component mainly carries low-frequency / global lighting
information.  Following the paper we adopt the classical U-Net architecture
to correct the brightness of low-light regions while keeping spatial
smoothness.  The network operates on the 3-channel illumination map
``L_low`` produced by the latent-space Retinex decomposition (resolution
H/8 x W/8) and outputs the enhanced illumination map ``L_enh``.

The ground-truth illumination ``L_gt`` used for supervision is obtained by
decomposing the normal-light reference image with the frozen decomposition
network.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBlock(nn.Module):
    """conv -> LeakyReLU -> conv -> LeakyReLU"""

    def __init__(self, in_ch, out_ch):
        super(ConvBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, 1, 1)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, 1, 1)
        self.relu = nn.LeakyReLU(0.1, inplace=True)

    def forward(self, x):
        return self.relu(self.conv2(self.relu(self.conv1(x))))


class EncBlock(nn.Module):
    """ConvBlock + 2x spatial downsampling"""

    def __init__(self, in_ch, out_ch):
        super(EncBlock, self).__init__()
        self.block = ConvBlock(in_ch, out_ch)
        self.down = nn.Conv2d(out_ch, out_ch, 3, 2, 1)

    def forward(self, x):
        feat = self.block(x)
        return feat, self.down(feat)


class DecBlock(nn.Module):
    """Upsample + skip-connection + ConvBlock.

    ``nn.Upsample(scale_factor=2)`` + conv is used so that feature maps of
    arbitrary (non power-of-two) sizes are handled correctly.
    """

    def __init__(self, in_ch, skip_ch, out_ch):
        super(DecBlock, self).__init__()
        self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False)
        self.conv = nn.Conv2d(in_ch, out_ch, 3, 1, 1)
        self.block = ConvBlock(out_ch + skip_ch, out_ch)

    def forward(self, x, skip):
        x = self.up(x)
        if x.shape[-2:] != skip.shape[-2:]:
            x = F.interpolate(x, size=skip.shape[-2:], mode="bilinear",
                              align_corners=False)
        x = self.conv(x)
        x = torch.cat([x, skip], dim=1)
        return self.block(x)


class IllumUNet(nn.Module):
    """Small U-Net for illumination enhancement (works at H/8 x W/8)."""

    def __init__(self, in_ch=3, out_ch=3, base_ch=32):
        super(IllumUNet, self).__init__()
        self.enc0 = ConvBlock(in_ch, base_ch)
        self.enc1 = EncBlock(base_ch, base_ch * 2)
        self.enc2 = EncBlock(base_ch * 2, base_ch * 4)

        self.bottleneck = ConvBlock(base_ch * 4, base_ch * 4)

        self.dec2 = DecBlock(base_ch * 4, base_ch * 4, base_ch * 2)
        self.dec1 = DecBlock(base_ch * 2, base_ch * 2, base_ch)
        self.dec0 = DecBlock(base_ch, base_ch, base_ch)

        self.head = nn.Sequential(
            nn.Conv2d(base_ch, base_ch, 3, 1, 1),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(base_ch, out_ch, 3, 1, 1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        f0 = self.enc0(x)                       # 1x
        f1, d1 = self.enc1(f0)                  # 1/2
        f2, d2 = self.enc2(d1)                  # 1/4

        y = self.bottleneck(d2)

        y = self.dec2(y, f2)                    # 1/2
        y = self.dec1(y, f1)                    # 1x
        y = self.dec0(y, f0)

        return self.head(y)


if __name__ == "__main__":
    net = IllumUNet()
    x = torch.rand(2, 3, 50, 75)
    with torch.no_grad():
        y = net(x)
    print("input :", x.shape)
    print("output:", y.shape)
