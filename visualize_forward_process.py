# %%
import matplotlib.pyplot as plt
import torch
import math
import numba
import numpy as np
import torch.nn.functional as F
from typing import List, Literal
from torchvision.io import read_image
import torchvision.transforms as transforms
from PIL import Image


@numba.jit(nopython=True, parallel=False)
def scatter(array, index, value):
    for (h, w), v in zip(index, value):
        array[h, w] = v
    return array


def load_points_as_images(
    point_path: str,
    scan_unfolding: bool = True,
    H: int = 64,
    W: int = 2048,
    min_depth: float = 1.45,
    max_depth: float = 80.0,
):
    """
        Transfers 3D point clouds to 2D intensity, depth images, mask, and point xyz values.
    """
    # load xyz & intensity and add depth & mask
    points = np.fromfile(point_path, dtype=np.float32).reshape((-1, 4))
    xyz = points[:, :3]  # xyz
    x = xyz[:, [0]]
    y = xyz[:, [1]]
    z = xyz[:, [2]]
    depth = np.linalg.norm(xyz, ord=2, axis=1, keepdims=True)
    mask = (depth >= min_depth) & (depth <= max_depth)
    points = np.concatenate([points, depth, mask], axis=1)

    if scan_unfolding:
        # the i-th quadrant
        # suppose the points are ordered counterclockwise
        quads = np.zeros_like(x, dtype=np.int32)
        quads[(x >= 0) & (y >= 0)] = 0  # 1st
        quads[(x < 0) & (y >= 0)] = 1  # 2nd
        quads[(x < 0) & (y < 0)] = 2  # 3rd
        quads[(x >= 0) & (y < 0)] = 3  # 4th

        # split between the 3rd and 1st quadrants
        diff = np.roll(quads, shift=1, axis=0) - quads
        delim_inds, _ = np.where(diff == 3)  # number of lines
        inds = list(delim_inds) + [len(points)]  # add the last index

        # vertical grid
        grid_h = np.zeros_like(x, dtype=np.int32)
        cur_ring_idx = H - 1  # ...0
        for i in reversed(range(len(delim_inds))):
            grid_h[inds[i] : inds[i + 1]] = cur_ring_idx
            if cur_ring_idx >= 0:
                cur_ring_idx -= 1
            else:
                break
    else:
        fup, fdown = np.deg2rad(3), np.deg2rad(-25)
        pitch = np.arcsin(z / depth) + abs(fdown)
        grid_h = 1 - pitch / (fup - fdown)
        grid_h = np.floor(grid_h * H).clip(0, H - 1).astype(np.int32)

    # horizontal grid
    yaw = -np.arctan2(y, x)  # [-pi,pi]
    grid_w = (yaw / np.pi + 1) / 2 % 1  # [0,1]
    grid_w = np.floor(grid_w * W).clip(0, W - 1).astype(np.int32)

    grid = np.concatenate((grid_h, grid_w), axis=1)

    # projection
    order = np.argsort(-depth.squeeze(1))
    proj_points = np.zeros((H, W, 4 + 2), dtype=points.dtype)
    proj_points = scatter(proj_points, grid[order], points[order])

    return proj_points.astype(np.float32)


def cosine_beta_schedule(steps, s=0.008):
    """
    cosine schedule
    as proposed in https://openreview.net/forum?id=-NEXDKk8gZ
    """
    t = torch.linspace(0, steps, steps + 1, dtype=torch.float64) / steps
    alphas_bar = torch.cos((t + s) / (1 + s) * math.pi * 0.5) ** 2
    alphas_bar = alphas_bar / alphas_bar[0]
    betas = 1 - (alphas_bar[1:] / alphas_bar[:-1])
    return torch.clip(betas, 0, 0.999)


def randn(
    *shape,
    rng: List[torch.Generator] | torch.Generator | None = None,
    **kwargs,
) -> torch.Tensor:
    if rng is None:
        return torch.randn(*shape, **kwargs)
    elif isinstance(rng, torch.Generator):
        return torch.randn(*shape, generator=rng, **kwargs)
    elif isinstance(rng, list):
        assert len(rng) == shape[0]
        return torch.stack(
            [torch.randn(*shape[1:], generator=r, **kwargs) for r in rng]
        )
    else:
        raise ValueError(f"invalid rng: {rng}")


def randn_like(
    x: torch.Tensor,
    rng: List[torch.Generator] | torch.Generator | None = None,
) -> torch.Tensor:
    return randn(*x.shape, rng=rng, device=x.device, dtype=x.dtype)


def sample_timesteps(self, batch_size: int) -> torch.Tensor:
    """
        Random sampling timestep
    """
    # discrete timesteps
    return torch.randint(
        low=0,
        high=1024,
        size=(batch_size,),
        dtype=torch.long,
    )


def q_sample(x0, steps, noise):
    """
        Get sample at specific timesteps
    """
    beta = cosine_beta_schedule(1024, s=0.008)
    beta = beta[:, None, None, None]
    alpha = 1 - beta
    alpha_bar = torch.cumprod(alpha, dim=0)
    alpha_bar = alpha_bar[steps]
    xt = alpha_bar.sqrt() * x0 + (1 - alpha_bar).sqrt() * noise
    return xt


def main():
    #### Forward diffusion process
    # path = "./data/kitti_360/dataset/data_3d_raw/2013_05_28_drive_0010_sync/velodyne_points/data/0000000000.bin"
    # image = torch.Tensor(load_points_as_images(path, W=512, H=64)[:, :, 3])
    # noise = randn_like(image)

    # ### This yields a random timestep ###
    # steps = sample_timesteps(image.shape[0], 1)
    # ### This allows to chooce specific timesteps ###
    # self_select_steps = torch.IntTensor([1])
    # xt = q_sample(image, self_select_steps, noise)

    # plt.imshow(xt[0, 0, :, :], interpolation="none")


    # timesteps = [0, 20, 80, 150, 275, 500, 1023]
    # fig, axes = plt.subplots(len(timesteps), 1)

    # for i, t in enumerate(timesteps):
    #     self_select_steps = torch.IntTensor([t])
    #     xt = q_sample(image, self_select_steps, noise)
    #     axes[i].set_axis_off()
    #     axes[i].imshow(xt[0, 0, :, :], interpolation="none")
    #### Forward diffusion process

    #### Diffusion process 
    path = "diffusion_process_photo.png"
    image = Image.open(path)

    transform = transforms.Compose([transforms.ToTensor(),])
    image = transform(image)
    # image = read_image(path)
    
    noise = randn_like(image.to(torch.float32))

    timesteps = [1023, 500, 200, 0]
    fig, axes = plt.subplots(1, len(timesteps), dpi=1000)

    # Visualize the gradual change in timesteps
    for i, t in enumerate(timesteps):
        self_select_steps = torch.IntTensor([t])
        xt = q_sample(image, self_select_steps, noise)
        axes[i].set_axis_off()
        axes[i].imshow(xt[0].permute(1,2,0), interpolation="none")
        # aspect_ratio = xt.shape[2] / xt.shape[3]
        # axes[i].set_aspect(aspect_ratio)
        axes[i].set_aspect("equal")


    plt.savefig("diffusion_process_plot.png", bbox_inches='tight', pad_inches=0)

    ####

if __name__ == "__main__":
    main()
