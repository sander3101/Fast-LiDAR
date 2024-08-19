# %%
import argparse
from pathlib import Path

import einops
import imageio
import matplotlib.cm as cm
import torch
import torch.nn.functional as F
from torchvision.utils import make_grid, save_image
from tqdm.auto import tqdm

import utils.inference
import utils.render
import numpy as np
import datasets as ds
from torch.utils.data import DataLoader
import random as rnd
from utils.lidar import LiDARUtility
import matplotlib.pyplot as plt
import torch
import os

checkpoint = "./logs/diffusion/kitti_360/spherical-1024/r2dm_only_quarter_upsample_continous/models/diffusion_0000300000.pth"

ddpm, lidar_utils, _ = utils.inference.setup_model(checkpoint, "cpu")


def render(xyz):
    """
        Transfers 2D intensity, depth images, mask, and point xyz values to a 3D point cloud.
    """
    xyz /= lidar_utils.max_depth
    z_min, z_max = -2 / lidar_utils.max_depth, 0.5 / lidar_utils.max_depth
    z = (xyz[:, [2]] - z_min) / (z_max - z_min)
    colors = utils.render.colorize(z.clamp(0, 1), cm.viridis) / 255
    R, t = utils.render.make_Rt(pitch=torch.pi / 3, yaw=torch.pi / 4, z=0.8)
    bev = 1 - utils.render.render_point_clouds(
        points=einops.rearrange(xyz, "B C H W -> B (H W) C"),
        colors=1 - einops.rearrange(colors, "B C H W -> B (H W) C"),
        R=R.to(xyz),
        t=t.to(xyz),
    )
    return bev


def check_equality(sample1, sample2):
    """
        Check that two files are the same.
        Used to debug for corrupt files
    """
    equality = torch.eq(sample1, sample2)

    counter1 = 0
    counter2 = 0
    for i in range(equality.shape[0]):
        for j in range(equality.shape[1]):
            if equality[i, j]:
                counter1 += 1
            else:
                counter2 += 1

    print(f"Equality rate: {(counter1 / counter2) * 100} %")
    print(counter1, counter2)


def change_lidargen_data_order(path):
    """
        LiDARGen samples have a different sampling id order on all samples, although the data is the same.
        This function converts the order to match the rest of the baseline sample id's.
    """
    with open("lidargen_test_paths.txt") as f:
        lidargen_test_paths = f.read().splitlines()

    indices = np.argsort(lidargen_test_paths)

    ours_dir = Path("sampling_results/quarter_sampling/all_feature_mixed/")
    base_dir = Path("lidargen_test_paths.txt")

    path_gt = list(sorted(ours_dir.glob("densification_targets/*.pth")))
    path_base = list(sorted(base_dir.glob("results/*.pth")))


def check_common_files(paths):
    """
        Check for common files in all models. 
        This is in case of corrupt files or lost files during transfering from servers or databases.
    """
    with open("lidargen_test_paths.txt") as f:
        lidargen_test_paths = f.read().splitlines()
        lidargen_test_paths = [
            lidar.split("/")[-1].split(".")[0] + ".pth" for lidar in lidargen_test_paths
        ]

    ext_paths = [f"{paths[i]}/results/" for i in range(len(paths))]
    files = [os.listdir(path) for path in ext_paths]

    elements_in_all = list(set.intersection(*map(set, files)))

    # lidargen = os.listdir(
    #     "baseline_results/baseline_results/diffusion_lidargen/results"
    # )

    # sorted_list = list(np.sort(lidargen))

    # val = elements_in_all[0]
    # index = sorted_list.index(val)
    # indices = np.argsort(lidargen_test_paths)

    # print(lidargen_test_paths.index("0000000000.pth"))

    # for i in range(len(elements_in_all)):
    #     path_gt = list(sorted(sorted_list))[i]
    #     path_base = list(sorted(lidargen_test_paths))[indices[i]]

    #     print(path_gt, path_base)

    # for path in ext_paths:
    #     for file in elements_in_all:
    #         sample = os.path.exists(f"{path}/{file}")
    #         if not sample:
    #             print("This file is wrong")

    # print(elements_in_all)


def main():
    path1 = "baseline_results/baseline_results/diffusion_r2dm/results/0000000000.pth"
    path2 = (
        "baseline_results/baseline_results/diffusion_lidargen/results/0000011478.pth"
    )

    sample1 = torch.load(path1)
    sample2 = torch.load(path2)

    check_equality(sample1[0], sample2[0])

    xyz = sample1[None, 1:4, :]

    bev = render(xyz)
    save_image(bev, "test.png")

    fig, axes = plt.subplots(2, 1)

    axes[0].imshow(sample1[0])

    axes[1].imshow(sample2[0])

    plt.show()

    ######### Check for common files in all models #########
    paths = [
        # "baseline_results/baseline_results/diffusion_lidargen",
        "baseline_results/baseline_results/diffusion_r2dm",
        "baseline_results/baseline_results/interp_bicubic",
        "baseline_results/baseline_results/interp_bilinear",
        "baseline_results/baseline_results/interp_nearest",
        "baseline_results/baseline_results/supervised_iln",
        "baseline_results/baseline_results/supervised_liif",
        "baseline_results/baseline_results/supervised_lsr",
    ]

    check_common_files(paths)
    ########################################################


if __name__ == "__main__":
    main()
