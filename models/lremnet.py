"""LREMNet: Internal Physical Prior-Guided Latent-Space Retinex Mamba Network.

Pipeline (paper Fig. 2, Sec. 3):

    1.  Latent-space Retinex decomposition (``models.decom.CTDN``):
        low-light image  ->  L_low, R_low   (H/8 x W/8)
    2.  Illumination enhancement branch (``models.illum.IllumUNet``):
        L_enh = IllumUNet(L_low)                               (Eq. 8)
    3.  Internal physical prior self-mining network:
        F_attn1 = CrossAttn(Q=L_enh, KV=R_low)                 (Eq. 9)
        F_mamba' = EdgeMamba(F_attn1)
        F_edge = Sobel(F_mamba'),  F_edge' = [F_mamba', F_edge]  (Eq. 12)
    4.  Reflectance enhancement guided by internal priors:
        CrossAttn(Q=F_edge', KV=R_low) -> EdgeMamba -> PiecesMamba -> R_enh
    5.  Fusion module:
        CrossAttn(Q=L_enh, KV=R_enh) -> EdgeMamba -> PiecesMamba -> decoder -> I_enh
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from models.decom import CTDN
from models.illum import IllumUNet
from models.CrossAttetion import Attention
from models.edgemamba import GradStateSpaceBlock, GradientExtractor
from models.piecesmamba import PatchMamba


def run_edge_mamba(blocks, feat):
    """Run a list of GradStateSpaceBlock on a (B, C, H, W) feature map."""
    B, C, H, W = feat.shape
    x = feat.contiguous().view(B, C, H * W).permute(0, 2, 1).contiguous()
    for blk in blocks:
        x = blk(x, (H, W))
    out = x.permute(0, 2, 1).view(B, C, H, W).contiguous()
    return out


class Decoder(nn.Module):
    """Decode fused 3-channel features (H/8 x W/8) to full-resolution image.

    Three pixel-shuffle stages give 8x upsampling (50x75 -> 400x600 for the
    paper's 400x600 training resolution).
    """

    def __init__(self, in_ch=3):
        super(Decoder, self).__init__()
        self.conv1 = nn.Sequential(nn.Conv2d(in_ch, 96, 3, 1, 1), nn.LeakyReLU(0.1, inplace=True))
        self.conv2 = nn.Sequential(nn.Conv2d(24, 48, 3, 1, 1), nn.LeakyReLU(0.1, inplace=True))
        self.conv3 = nn.Sequential(nn.Conv2d(12, 24, 3, 1, 1), nn.LeakyReLU(0.1, inplace=True))
        self.ps1 = nn.PixelShuffle(2)
        self.ps2 = nn.PixelShuffle(2)
        self.ps3 = nn.PixelShuffle(2)
        self.conv_out = nn.Conv2d(6, 3, 3, 1, 1)

        # zero-init the final layer: the initial prediction is therefore the
        # (bilinearly upsampled) low-light input, which stabilises training
        nn.init.zeros_(self.conv_out.weight)
        nn.init.zeros_(self.conv_out.bias)

    def forward(self, x):
        x = self.ps1(self.conv1(x))
        x = self.ps2(self.conv2(x))
        x = self.ps3(self.conv3(x))
        return self.conv_out(x)


class LREMNet(nn.Module):
    """Full LREMNet model.

    Args:
        channels: base channel width of the enhancement branches.
        num_edge_blocks: number of GradStateSpaceBlock inside each EdgeMamba.
        pieces_num_blocks: number of local patches for PiecesMamba (paper: 12).
        pieces_layers: number of SS2D + FF layers inside PiecesMamba.
    """

    def __init__(self, channels=64, num_edge_blocks=2,
                 pieces_num_blocks=12, pieces_layers=2):
        super(LREMNet, self).__init__()
        c = channels

        # ---------- 1. latent Retinex decomposition (frozen in stage 2) -----
        self.decom = CTDN(channels)

        # ---------- 2. illumination enhancement branch ----------------------
        self.illum_net = IllumUNet(in_ch=3, out_ch=3, base_ch=32)

        # ---------- 3. internal physical prior self-mining ------------------
        self.proj_l1 = nn.Conv2d(3, c, 3, 1, 1)          # L_enh  -> Q
        self.proj_r1 = nn.Conv2d(3, c, 3, 1, 1)          # R_low  -> K, V
        self.cross_attn1 = Attention(dim=c, num_heads=8, bias=False)
        self.edge_mamba1 = nn.ModuleList(
            [GradStateSpaceBlock(dim=c, d_state=16) for _ in range(num_edge_blocks)])
        self.grad_extractor = GradientExtractor()        # (B, C, H, W) -> (B, 1, H, W)
        self.edge_fuse = nn.Conv2d(c + 1, c, 3, 1, 1)    # F_edge' = [F_mamba', F_edge]

        # ---------- 4. reflectance enhancement ------------------------------
        self.proj_r2 = nn.Conv2d(3, c, 3, 1, 1)          # R_low  -> K, V
        self.proj_edge = nn.Conv2d(c, c, 3, 1, 1)        # F_edge' -> Q
        self.cross_attn2 = Attention(dim=c, num_heads=8, bias=False)
        self.edge_mamba2 = nn.ModuleList(
            [GradStateSpaceBlock(dim=c, d_state=16) for _ in range(num_edge_blocks)])
        self.to_r = nn.Conv2d(c, 3, 3, 1, 1)
        self.pieces_reflect = PatchMamba(input_channels=3,
                                         num_blocks=pieces_num_blocks,
                                         num_mamba_layers=pieces_layers)

        # ---------- 5. fusion module ----------------------------------------
        self.proj_l3 = nn.Conv2d(3, c, 3, 1, 1)          # L_enh  -> Q
        self.proj_r3 = nn.Conv2d(3, c, 3, 1, 1)          # R_enh  -> K, V
        self.cross_attn3 = Attention(dim=c, num_heads=8, bias=False)
        self.edge_mamba3 = nn.ModuleList(
            [GradStateSpaceBlock(dim=c, d_state=16) for _ in range(num_edge_blocks)])
        self.to_img = nn.Conv2d(c, 3, 3, 1, 1)
        self.pieces_fuse = PatchMamba(input_channels=3,
                                      num_blocks=pieces_num_blocks,
                                      num_mamba_layers=pieces_layers)
        self.decoder = Decoder(in_ch=3)

    # ------------------------------------------------------------------ #
    # helpers
    # ------------------------------------------------------------------ #
    @torch.no_grad()
    def decompose_low(self, low_img):
        """Run the frozen decomposition network on a low-light image."""
        x = torch.cat([low_img, low_img], dim=1)
        out = self.decom(x, pred_fea=None)
        return out["low_R"], out["low_L"]

    @torch.no_grad()
    def decompose_high(self, high_img):
        """Run the frozen decomposition network on a normal-light image."""
        x = torch.cat([high_img, high_img], dim=1)
        out = self.decom(x, pred_fea=None)
        return out["high_R"], out["high_L"]

    def forward(self, low_img):
        """Enhance a low-light image.

        Args:
            low_img: (B, 3, H, W) in [0, 1], H and W should be multiples of 8.

        Returns dict with keys:
            L_low, R_low, L_enh, R_enh, F_edge, I_enh
        """
        B, _, H, W = low_img.shape

        # 1. latent-space Retinex decomposition
        R_low, L_low = self.decompose_low(low_img)          # (B, 3, H/8, W/8)

        # 2. illumination enhancement
        L_enh = self.illum_net(L_low)                       # (B, 3, H/8, W/8)

        # 3. internal physical prior self-mining
        q1 = self.proj_l1(L_enh)
        kv1 = self.proj_r1(R_low)
        F_attn1 = kv1 + self.cross_attn1(kv1, q1)           # Eq. 9
        F_mamba = run_edge_mamba(self.edge_mamba1, F_attn1)  # (B, C, H/8, W/8)

        F_edge = self.grad_extractor(F_mamba)               # (B, 1, H/8, W/8)
        F_edge_cat = self.edge_fuse(torch.cat([F_mamba, F_edge], dim=1))  # F_edge'

        # 4. reflectance enhancement guided by internal priors
        q2 = self.proj_edge(F_edge_cat)
        kv2 = self.proj_r2(R_low)
        F_ref = kv2 + self.cross_attn2(kv2, q2)
        F_ref = run_edge_mamba(self.edge_mamba2, F_ref)
        R_enh = self.pieces_reflect(self.to_r(F_ref))       # (B, 3, H/8, W/8)

        # 5. fusion
        q3 = self.proj_l3(L_enh)
        kv3 = self.proj_r3(R_enh)
        F_fuse = kv3 + self.cross_attn3(kv3, q3)
        F_fuse = run_edge_mamba(self.edge_mamba3, F_fuse)
        F_fuse = self.pieces_fuse(self.to_img(F_fuse))      # (B, 3, H/8, W/8)

        I_enh = self.decoder(F_fuse)                        # (B, 3, H, W)
        I_enh = I_enh + F.interpolate(low_img, size=(H, W), mode="bilinear",
                                      align_corners=False)

        return {
            "L_low": L_low,
            "R_low": R_low,
            "L_enh": L_enh,
            "R_enh": R_enh,
            "F_edge": F_edge,
            "I_enh": I_enh,
        }


if __name__ == "__main__":
    torch.manual_seed(0)
    net = LREMNet(channels=64)
    net.eval()
    x = torch.rand(1, 3, 96, 96)
    with torch.no_grad():
        out = net(x)
    for k, v in out.items():
        print(f"{k:6s}: {tuple(v.shape)}")
    n_params = sum(p.numel() for p in net.parameters() if p.requires_grad)
    print(f"trainable params: {n_params / 1e6:.2f} M")
