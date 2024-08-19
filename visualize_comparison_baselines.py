# %%

from pathlib import Path

import einops
import matplotlib.cm as cm
import matplotlib.pyplot as plt
import torch
from utils.render import colorize, make_Rt, render_point_clouds

# Filepath name for the results of all baselines for comparison
methods = [
    "interp_nearest",
    "interp_bilinear",
    "interp_bicubic",
    "supervised_lsr",
    "supervised_iln",
    "diffusion_lidargen_fixed",
    "diffusion_ours_sander",
    "diffusion_r2dm",
]

# Transforming baseline names for more readable results on plot
translation = {
    "interp_nearest": "Nearest neighbor",
    "interp_bilinear": "Bilinear",
    "interp_bicubic": "Bicubic",
    "supervised_lsr": "LSR",
    "supervised_iln": "ILN",
    "diffusion_lidargen_fixed": "LiDARGen",
    "diffusion_r2dm": "R2DM",
    "diffusion_ours_sander": "Ours",
    "gt": "GT",
    "input": "Input",
}


def render_xyz(xyz, max_depth=80.0):
    """
        Transfers 2D intensity, depth images, mask, and point xyz values to a 3D point cloud.
    """
    z_min, z_max = -2 / max_depth, 0.5 / max_depth
    z = (xyz[:, [2]] - z_min) / (z_max - z_min)
    colors = colorize(z.clamp(0, 1), cm.viridis) / 255
    points = einops.rearrange(xyz, "B C H W -> B (H W) C") / max_depth
    colors = 1 - einops.rearrange(colors, "B C H W -> B (H W) C")
    R, t = make_Rt(pitch=torch.pi / 4, yaw=torch.pi / 4, z=0.6, device=xyz.device)
    bev = 1 - render_point_clouds(points=points, colors=colors, R=R, t=t)
    bev = einops.rearrange(bev, "B C H W -> B H W C")
    return bev


def redner_img(img):
    """
        Renders image for visualization
    """
    img = colorize(img)
    img = einops.rearrange(img, "B C H W -> B H W C")
    return img


def parse_data(index, root):
    """
        Parse data from all baselines
    """
    root = Path(root)

    # prediction
    for method in methods:
        method_dir = sorted((root / method).glob("**/upsample_4x"))[0]
        pred_path = sorted(method_dir.glob("results/*.pth"))[index]
        tensor = torch.load(pred_path, map_location="cpu")  # (5,H,W)
        d, xyz, r = tensor.split([1, 3, 1], dim=0)
        yield method, d, r, xyz

    # ground truth
    gt_path = str(pred_path).replace("results", "targets")
    tensor = torch.load(gt_path, map_location="cpu")  # (5,H,W)
    d, xyz, r = tensor.split([1, 3, 1], dim=0)
    yield "gt", d, r, xyz

    # input
    mask = torch.zeros_like(tensor)
    mask[:, ::4] = 1.0
    input = tensor * mask
    d, xyz, r = input.split([1, 3, 1], dim=0)
    yield "input", d, r, xyz


packed = parse_data(index=1949, root="/media/sander/lab/R2DM")
methods, Ds, Rs, XYZs = zip(*packed)

Ds = torch.stack(Ds)
Rs = torch.stack(Rs)
XYZs = torch.stack(XYZs)

Ds = redner_img(Ds / 80.0)  # (B,H,W,3)
Rs = redner_img(Rs)  # (B,H,W,3)
BEVs = render_xyz(XYZs)  # (B,H,W,3)

# Plots all baseline results in a grid
for i, method in enumerate(methods):
    fig, ax = plt.subplots(
        3,
        1,
        figsize=(10, 10),
        gridspec_kw={"height_ratios": [100, 30, 30]},
        constrained_layout=True,
    )
    # ax[0].set_title(translation[method])
    ax[0].imshow(BEVs[i])
    ax[1].imshow(Ds[i, :, :256], interpolation="none")
    ax[2].imshow(Rs[i, :, :256], interpolation="none")
    [a.axis("off") for a in ax.ravel()]
    plt.savefig(
        f"results_{method}.pdf",
        bbox_inches="tight",
        pad_inches=0.0,
        dpi=300,
    )
