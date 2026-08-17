"""Loss functions of LREMNet.

Stage 1 (latent-space Retinex decomposition, paper Sec. 3.1):
    * reconstruction loss:       ||R * L - fea||_1
    * reflectance consistency:   ||R_low - R_high||_1
    * illumination smoothness:   gradient-weighted TV on L
    * illumination regularizer:  mean(L) ~ mean brightness of fea

Stage 2 (full enhancement network, paper Eq. 13 - 17):
    L_total = 0.5 * L_ill + 0.5 * L_edge + 0.5 * L_color + 1.0 * L_rec
    * L_ill   = ||L_enh - L_gt||_1                     (Eq. 14)
    * L_edge  = ||F_edge - grad(I_gt)||_1              (Eq. 15)
    * L_color = sum_c angle(I_enh^c, I_gt^c)           (Eq. 16)
    * L_rec   = ||I_enh - I_gt||_1                     (Eq. 17)
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


# --------------------------------------------------------------------- #
# Stage 1 decomposition losses
# --------------------------------------------------------------------- #
def illumination_smooth_loss(illumination, image, lambda_grad=10.0):
    """Gradient-weighted TV loss.

    Edges of the guiding image are allowed to have strong illumination
    gradients; flat regions are penalised.
    """
    grad_l_x = torch.abs(illumination[:, :, :, 1:] - illumination[:, :, :, :-1])
    grad_l_y = torch.abs(illumination[:, :, 1:, :] - illumination[:, :, :-1, :])

    if image.shape[1] == 3:
        gray = 0.299 * image[:, 0:1] + 0.587 * image[:, 1:2] + 0.114 * image[:, 2:3]
    else:
        gray = image

    grad_i_x = torch.abs(gray[:, :, :, 1:] - gray[:, :, :, :-1])
    grad_i_y = torch.abs(gray[:, :, 1:, :] - gray[:, :, :-1, :])

    w_x = torch.exp(-lambda_grad * grad_i_x)
    w_y = torch.exp(-lambda_grad * grad_i_y)

    return (grad_l_x * w_x).mean() + (grad_l_y * w_y).mean()


def decomposition_loss(output, w_recon=1.0, w_reflectance=1.0,
                       w_smooth=0.01, w_illum=0.1):
    """Stage-1 loss for the latent Retinex decomposition network.

    Args:
        output: dict from CTDN with keys low_R, low_L, low_fea,
                high_R, high_L, high_fea.
    """
    l1 = nn.L1Loss()
    l2 = nn.MSELoss()

    low_R, low_L, low_fea = output["low_R"], output["low_L"], output["low_fea"]
    high_R, high_L, high_fea = output["high_R"], output["high_L"], output["high_fea"]

    # feature-space reconstruction: R * L ~ low_fea
    loss_recon = (l1(low_R * low_L, low_fea) + l1(high_R * high_L, high_fea)) * w_recon

    # reflectance is lighting-invariant: low_R ~ high_R
    loss_reflectance = l1(low_R, high_R) * w_reflectance

    # illumination smoothness, guided by the compressed feature map
    loss_smooth = (illumination_smooth_loss(low_L, low_fea)
                   + illumination_smooth_loss(high_L, high_fea)) * w_smooth

    # illumination mean ~ feature brightness
    low_gray = 0.299 * low_fea[:, 0:1] + 0.587 * low_fea[:, 1:2] + 0.114 * low_fea[:, 2:3]
    loss_illum = l2(low_L[:, 0:1].mean(dim=[1, 2, 3]),
                    low_gray.mean(dim=[1, 2, 3])) * w_illum

    total = loss_recon + loss_reflectance + loss_smooth + loss_illum
    return {
        "total": total,
        "recon": loss_recon,
        "reflectance": loss_reflectance,
        "smooth": loss_smooth,
        "illum": loss_illum,
    }


# --------------------------------------------------------------------- #
# Stage 2 enhancement losses
# --------------------------------------------------------------------- #
def gradient_magnitude(img):
    """Sobel-based gradient magnitude of a gray image, shape (B, 1, H, W)."""
    sobel_x = torch.tensor([[-1., 0., 1.], [-2., 0., 2.], [-1., 0., 1.]],
                           device=img.device).view(1, 1, 3, 3)
    sobel_y = torch.tensor([[-1., -2., -1.], [0., 0., 0.], [1., 2., 1.]],
                           device=img.device).view(1, 1, 3, 3)
    if img.shape[1] == 3:
        gray = 0.299 * img[:, 0:1] + 0.587 * img[:, 1:2] + 0.114 * img[:, 2:3]
    else:
        gray = img
    gx = F.conv2d(gray, sobel_x, padding=1)
    gy = F.conv2d(gray, sobel_y, padding=1)
    return torch.sqrt(gx ** 2 + gy ** 2 + 1e-6)


def color_angle_loss(pred, target):
    """Sum over RGB channels of the radian angle between flattened vectors.

    angle(a, b) = arccos(<a, b> / (||a|| ||b||)), averaged over pixels.
    """
    loss = 0.0
    for c in range(3):
        a = pred[:, c:c + 1].flatten(1)
        b = target[:, c:c + 1].flatten(1)
        cos = F.cosine_similarity(a, b, dim=-1, eps=1e-8).clamp(-1.0 + 1e-7, 1.0 - 1e-7)
        loss = loss + torch.acos(cos).mean()
    return loss


def enhancement_loss(output, high_img, L_gt, w_ill=0.5, w_edge=0.5,
                     w_color=0.5, w_rec=1.0):
    """Stage-2 total loss (paper Eq. 13).

    Args:
        output: dict from LREMNet.forward (L_enh, F_edge, I_enh).
        high_img: normal-light reference (B, 3, H, W).
        L_gt: ground-truth illumination from decomposing high_img (B, 3, H/8, W/8).
    """
    l1 = nn.L1Loss()
    L_enh, F_edge, I_enh = output["L_enh"], output["F_edge"], output["I_enh"]

    # illumination reconstruction (Eq. 14)
    loss_ill = l1(L_enh, L_gt)

    # gradient-guided loss (Eq. 15): internal edge prior vs GT gradient
    grad_gt = gradient_magnitude(high_img)                       # (B, 1, H, W)
    grad_gt = F.interpolate(grad_gt, size=F_edge.shape[-2:],
                            mode="bilinear", align_corners=False)
    loss_edge = l1(F_edge, grad_gt)

    # color consistency loss (Eq. 16)
    loss_color = color_angle_loss(I_enh, high_img)

    # image reconstruction loss (Eq. 17)
    loss_rec = l1(I_enh, high_img)

    total = (w_ill * loss_ill + w_edge * loss_edge
             + w_color * loss_color + w_rec * loss_rec)
    return {
        "total": total,
        "ill": loss_ill,
        "edge": loss_edge,
        "color": loss_color,
        "rec": loss_rec,
    }


if __name__ == "__main__":
    out = {
        "L_enh": torch.rand(2, 3, 16, 16),
        "F_edge": torch.rand(2, 1, 16, 16),
        "I_enh": torch.rand(2, 3, 128, 128),
    }
    high = torch.rand(2, 3, 128, 128)
    L_gt = torch.rand(2, 3, 16, 16)
    losses = enhancement_loss(out, high, L_gt)
    for k, v in losses.items():
        print(k, float(v))
