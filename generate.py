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



def upsampling(img, loss_level=4):
    """
        Generates upsampling mask
    """
    image = img[0, :, :]
    N, M = image.shape
    mask = np.zeros((N, M))

    for i in range(N):
        if i % loss_level != 0:
            mask[i, :] = 1

    return mask


def add_sparsity(img, sparsity_level=0.1):
    """
        Generates simple line mask
    """
    image = img[0, :, :]
    N, M = image.shape
    mask = np.zeros((N, M))

    for i in range(N):
        mask[i, :] = rnd.random() < sparsity_level

    return mask


def add_pepper(img, sparsity_level=0.1):
    """
        Generates pepper noise mask
    """
    image = img[0, :, :]
    N, M = image.shape
    mask = np.zeros((N, M))

    for i in range(N):
        for j in range(M):
            mask[i, j] = rnd.random() < sparsity_level

    return mask


def add_complex_sparsity(
    img,
    sparsity_level=0.1,
    start_row=0,
    stop_row=None,  # None, mean we use the full length N
    start_line=0,
    stop_line=None,  # None, mean we use the full length M
    up_prob=0.1,
    down_prob=0.1,
    thickness=1,
):
    """
    Adds noicy sparsity mask lines to an image

    Args:
        img (np.array) : image
        sparsity_levels (float): Percentage of sparsity level
        start_row (int): start row of mask
        stop_row (int): stop row of mask
        start_line (int): start line of mask
        stop_line (int): stop line of mask
        up_prob (float): Probability of line inclining
        down_prob (float): Probability of line declining
        thickness (int): thickness of mask line

    Return:
        mask (numpy.array): binary masked area
    """

    image = img[0, :, :]
    N, M = image.shape
    mask = np.zeros((N, M))

    # Define row stopping point for noise
    stop_row = N if stop_row == None else stop_row
    # Define line stopping point for noise
    stop_line = M if stop_line == None else stop_line
    # Make sure the thickness of the line can't be smaller than 1
    thickness = 1 if thickness <= 0 else thickness

    for i in range(start_row, stop_row):
        # Probability of adding sparsity
        if rnd.random() < sparsity_level:
            # Counter to controll incline and decline of the lines
            height_shift = 0
            changed_counter = 0

            for j in range(start_line, stop_line):
                """
                We use the formula 1/(log(|height_shift|)+1),
                in order to controll the amount incline and decline.
                The probability of continous incline and decline quickly declines,
                making it unprobable for a line to continue in one direction.
                """
                # The probability of change decreases for each change
                if rnd.random() < 1 / (abs(changed_counter) + 1):
                    if changed_counter < 5:
                        changed_counter += 1
                    else:
                        changed_counter = 0

                    if height_shift < 0:  # If the line is declining
                        if rnd.random() > 10 * up_prob / (
                            np.log(abs(height_shift)) + 1
                        ):
                            height_shift += 1
                        else:
                            height_shift -= 1

                    if height_shift > 0:  # If the line is inclining
                        if rnd.random() > 10 * down_prob / (
                            np.log(abs(height_shift)) + 1
                        ):
                            height_shift -= 1
                        else:
                            height_shift += 1

                # If the coordinate is still inbounds we mask the pixel
                if (
                    i + height_shift + thickness < N
                    and i + height_shift - thickness >= 0
                ):
                    mask[
                        i + height_shift - thickness : i + height_shift + thickness,
                        j,
                    ] = 1

                # 1/3 chance of going straight, incline or decline
                if height_shift == 0:
                    randnum = rnd.random()
                    if randnum < 0.3:  # Declining
                        height_shift -= 1
                    elif randnum > 0.7:  # Inclining
                        height_shift += 1

    return mask


def get_inpainting_mask(args, coords):
    """
        Generates the desired conditional mask for conditional generating
    """

    # Defines dataset
    dataset = ds.load_dataset(
        path=f"data/kitti_360",
        name="spherical-1024",
        split=ds.Split.TEST,
        num_proc=4,
    ).with_format("torch")

    print(len(dataset))

    # Create dataloader
    dataloader = DataLoader(
        dataset,
        batch_size=1,
        shuffle=True,
        num_workers=4,
        drop_last=True,
        pin_memory=True,
    )

    # Generates LiDAR utilities used for transformation and normalization
    lidar_utils = LiDARUtility(
        resolution=(64, 1024),
        image_format="log_depth",
        min_depth=1.45,
        max_depth=80.0,
        ray_angles=coords,
    )
    lidar_utils.to(args.device)

    if args.sample_id == -1:
        args.sample_id = rnd.randint(0, len(dataset))
    print(f"sample id: {args.sample_id}")

    batch = dataset[args.sample_id]

    x = []
    x += [lidar_utils.convert_depth(batch["depth"][None])]
    x += [batch["reflectance"][None]]
    x = torch.cat(x, dim=1)
    x = lidar_utils.normalize(x)
    x = F.interpolate(
        x.to(args.device),
        size=(64, 1024),
        mode="nearest-exact",
    )

    # Defines the sparsity level to be 10%-30% if not defined
    if args.sparsity is None:
        sparsity = rnd.uniform(0.1, 0.3)
    else:
        sparsity = args.sparsity

    if args.diffusion_mode == "simple": # If simple lines mask
        mask = torch.stack(
            [
                torch.tensor(add_sparsity(element, sparsity_level=sparsity))
                for element in x
            ]
        )

    elif args.diffusion_mode == "complex": # If complex lines mask
        mask = torch.stack(
            [
                torch.tensor(add_complex_sparsity(element, sparsity_level=sparsity))
                for element in x
            ]
        )
    elif args.diffusion_mode == "pepper": # If pepper noise mask
        mask = torch.stack(
            [
                torch.tensor(add_pepper(element, sparsity_level=sparsity))
                for element in x
            ]
        )
    elif args.diffusion_mode == "upsample": # If upsampling mask
        mask = torch.stack([torch.tensor(upsampling(element)) for element in x])
    elif args.diffusion_mode == "mixed": # If mixed training tasks
        random_select = rnd.random()
        if random_select >= 0.7:
            N, M = x[0, 0, :, :].shape

            mask = torch.stack(
                [
                    torch.tensor(
                        add_complex_sparsity(
                            element,
                            sparsity_level=rnd.uniform(0.1, 0.3),
                            start_row=rnd.randint(0, N // 2),
                            stop_row=rnd.randint(N // 2, N - 1),
                            start_line=rnd.randint(0, M // 2),
                            stop_line=rnd.randint(M // 2, M - 1),
                            up_prob=rnd.uniform(0.1, 0.2),
                            down_prob=rnd.uniform(0.1, 0.2),
                            thickness=rnd.randint(1, 2),
                        )
                    )
                    for element in x
                ]
            )
        elif random_select >= 0.5 and random_select < 0.7:
            mask = torch.stack([torch.tensor(upsampling(element)) for element in x])
        elif random_select >= 0.3 and random_select < 0.5:
            mask = torch.stack(
                [
                    torch.tensor(
                        add_pepper(element, sparsity_level=rnd.uniform(0.1, 0.5))
                    )
                    for element in x
                ]
            )
        else:
            mask = torch.stack(
                [
                    torch.tensor(
                        add_sparsity(element, sparsity_level=rnd.uniform(0.1, 0.5))
                    )
                    for element in x
                ]
            )

    return x, mask.unsqueeze(1).to(args.device)


def main(args):
    torch.set_grad_enabled(False)
    torch.backends.cudnn.benchmark = True

    # =================================================================================
    # Load pre-trained model
    # =================================================================================

    ddpm, lidar_utils, _ = utils.inference.setup_model(args.ckpt, device=args.device)

    # =================================================================================
    # Sampling (reverse diffusion)
    # =================================================================================


    sample_x, mask = get_inpainting_mask(args, ddpm.denoiser.coords)

    # Defines the conditional sampling function
    xs = ddpm.conditional_sample(
        batch_size=1,
        mode="ddpm",
        num_steps=args.sampling_steps,
        return_all=True,
        x_0=sample_x,
        mask=mask.float(),
    ).clamp(-1, 1)

    # =================================================================================
    # Save as image or video
    # =================================================================================

    xs = lidar_utils.denormalize(xs)
    xs[:, :, [0]] = lidar_utils.revert_depth(xs[:, :, [0]]) / lidar_utils.max_depth

    def render(x):
        """
            Render image and point cloud of input sample x
        """
        img = einops.rearrange(x, "B C H W -> B 1 (C H) W")
        img = utils.render.colorize(img) / 255
        xyz = lidar_utils.to_xyz(x[:, [0]] * lidar_utils.max_depth)
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
        return img, bev

    ########### Generates several different masks used for visual purposes in thesis illustrations and for testing ###########
    orig = lidar_utils.denormalize(sample_x)
    orig[:, [0]] = lidar_utils.revert_depth(orig[:, [0]]) / lidar_utils.max_depth

    orig = orig * (1 - mask).float()

    _, masked_bev = render(orig)
    save_image(
        masked_bev,
        f"masked_bev_steps_{args.sampling_steps}_id_{args.sample_id}.png",
        nrow=4,
    )

    img, bev = render(xs[-1])

    ##### For illustration
    save_image(img, "x_0_original.png", nrow=2)
    save_image(mask, "mask.png", nrow=2)

    save_image(
        img[0, :, :] * (1 - torch.cat([mask[0], mask[0]], dim=1).float()),
        "x0_time_mask.png",
        nrow=2,
    )

    save_image(
        img[0, :, :] * torch.cat([mask[0], mask[0]], dim=1).float(),
        f"x_time_mask.png",
        nrow=2,
    )
    #####

    # print(img[0, 0, 0:64, :].shape)
    save_image(
        img, f"samples_img_steps_{args.sampling_steps}_id_{args.sample_id}.png", nrow=1
    )
    save_image(
        bev, f"samples_bev_steps_{args.sampling_steps}_id_{args.sample_id}.png", nrow=4
    )
    save_image(
        img[0, :, :] * (1 - torch.cat([mask[0], mask[0]], dim=1).float()),
        # img[0, :, 0:64, :],
        f"sampling_mask_steps_{args.sampling_steps}_id_{args.sample_id}.png",
        nrow=2,
    )

    ###########################################################################################################################

    # Generates sampling video of conditional sampling
    video = imageio.get_writer("samples.mp4", mode="I", fps=60)
    for x in tqdm(xs, desc="making video..."):
        img, bev = render(x)
        scale = 512 / img.shape[-1]
        img = F.interpolate(img, scale_factor=scale, mode="bilinear", antialias=True)
        scale = 512 / bev.shape[-1]
        bev = F.interpolate(bev, scale_factor=scale, mode="bilinear", antialias=True)
        img = torch.cat([img, bev], dim=2)
        img = make_grid(img, nrow=args.batch_size, pad_value=1)
        img = img.permute(1, 2, 0).mul(255).byte()
        video.append_data(img.cpu().numpy())
    video.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", type=Path, required=True)
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cuda")
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--sampling_steps", type=int, default=256)
    parser.add_argument("--sparsity", type=float, default=None)
    parser.add_argument(
        "--diffusion_mode",
        choices=["simple", "complex", "pepper", "upsample", "mixed"],
        default="simple",
    )
    parser.add_argument("--sample_id", type=int, default=-1)
    args = parser.parse_args()
    args.device = torch.device(args.device)
    main(args)
