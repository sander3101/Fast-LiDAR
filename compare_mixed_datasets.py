# %%

from pathlib import Path

import einops
import matplotlib.cm as cm
import matplotlib.pyplot as plt
import torch
import os

from utils.render import colorize, make_Rt, render_point_clouds

# Filepaths for the different models trained on various datasets
methods = [
    "only_quarter_upsample",
    "only_quarter_upsample_all_datasets_kitti360",
    "only_quarter_upsample_all_datasets_kitti_raw",
    "only_quarter_upsample_all_datasets_synlidar",
    "only_quarter_upsample_tested_on_kitti_raw",
    "only_quarter_upsample_tested_on_synlidar",
]

# Method names for matching with thesis names
method_names = [
    "Config C",
    "Config I",
    "Config II",
    "Config III",
    "Config IV",
    "Config V",
]


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

    # Get the shortest of the datasets so that we don't risk index out of range.
    elements = os.listdir(
        f"sampling_results2/quarter_sampling/{methods[2]}/densification_results_resamplesteps_8/"
    )
    # Sort elements to get same index for all methods as their datasets varie in order
    elements = sorted(elements)

    # prediction
    for method in methods:
        if method == "only_quarter_upsample_all_datasets_synlidar":
            pred_path = f"sampling_results/quarter_sampling/{method}/densification_results_resamplesteps_8/{elements[index]}"

            tensor = torch.load(pred_path, map_location="cpu")  # (5,H,W)
            d, xyz, r = tensor.split([1, 3, 1], dim=0)
        else:
            pred_path = f"{root}/{method}/densification_results_resamplesteps_8/{elements[index]}"

            tensor = torch.load(pred_path, map_location="cpu")  # (5,H,W)
            d, xyz, r = tensor.split([1, 3, 1], dim=0)
        yield method, d, r, xyz


def main():
    sample_index = 50
    packed = parse_data(index=sample_index, root="sampling_results2/quarter_sampling")
    methods, Ds, Rs, XYZs = zip(*packed)

    Ds = torch.stack(Ds)
    Rs = torch.stack(Rs)
    XYZs = torch.stack(XYZs)

    Ds = redner_img(Ds / 80.0)  # (B,H,W,3)
    Rs = redner_img(Rs)  # (B,H,W,3)
    BEVs = render_xyz(XYZs)  # (B,H,W,3)

    fig, ax = plt.subplots(
        9,
        2,
        figsize=(8, 15),
        gridspec_kw={"height_ratios": [100, 30, 30, 100, 30, 30, 100, 30, 30]},
        constrained_layout=True,
    )

    # Plots all model results in a 2x3 grid
    for i in range(len(methods) // 3):
        ax[0][i].set_title(method_names[i].replace("_", " "), fontsize=30)
        ax[0][i].imshow(BEVs[i])
        ax[1][i].imshow(Ds[i, :, 256 * 2 : 256 * 3], interpolation="none")
        ax[2][i].imshow(Rs[i, :, 256 * 2 : 256 * 3], interpolation="none")
        ax[3][i].set_title(
            method_names[i + (len(methods) // 3)].replace("_", " "), fontsize=30
        )
        ax[3][i].imshow(BEVs[i + (len(methods) // 3)])
        ax[4][i].imshow(
            Ds[i + (len(methods) // 3), :, 256 * 2 : 256 * 3], interpolation="none"
        )
        ax[5][i].imshow(
            Rs[i + (len(methods) // 3), :, 256 * 2 : 256 * 3], interpolation="none"
        )
        ax[6][i].set_title(
            method_names[i + 2 * (len(methods) // 3)].replace("_", " "), fontsize=30
        )
        ax[6][i].imshow(BEVs[i + 2 * (len(methods) // 3)])
        ax[7][i].imshow(
            Ds[i + 2 * (len(methods) // 3), :, 256 * 2 : 256 * 3], interpolation="none"
        )
        ax[8][i].imshow(
            Rs[i + 2 * (len(methods) // 3), :, 256 * 2 : 256 * 3], interpolation="none"
        )

    # ax[6][0].set_title(method_names[len(methods) - 1].replace("_", " "), fontsize=30)
    # ax[6][0].imshow(BEVs[len(methods) - 1])
    # ax[7][0].imshow(Ds[len(methods) - 1, :, 256 * 2 : 256 * 3], interpolation="none")
    # ax[8][0].imshow(Rs[len(methods) - 1, :, 256 * 2 : 256 * 3], interpolation="none")

    [a.axis("off") for a in ax.ravel()]
    plt.savefig(
        "mixed_datasets_comparison.pdf", bbox_inches="tight", pad_inches=0, dpi=500
    )
    plt.show()


if __name__ == "__main__":
    main()

# %%
