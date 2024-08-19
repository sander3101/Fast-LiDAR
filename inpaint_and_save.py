import os
import warnings
from argparse import ArgumentParser
from pathlib import Path

import datasets as ds
import torch
from accelerate import Accelerator
from rich import print
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
import numpy as np
import random as rnd


import utils.inference

warnings.filterwarnings("ignore", category=UserWarning)


def add_complex_sparsity(
    img,
    sparsity_level=0.1,
    start_row=0,
    stop_row=None,  # None mean we use the full length N
    start_line=0,
    stop_line=None,  # None mean we use the full length M
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


def add_sparsity(img, sparsity_level=0.1):
    """
        Generates simple lines mask
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


def half_loss(img):
    """
        Generates a 2x upsampling mask
    """
    image = img[0, :, :]
    N, M = image.shape
    mask = np.zeros((N, M))

    for i in range(0, N, 2):
        mask[i, :] = 1

    return mask


def quarter_loss(img):
    """
        Generates a 4x upsampling mask
    """
    image = img[0, :, :]
    N, M = image.shape
    mask = np.zeros((N, M))

    for i in range(N):
        if i % 4 != 0:
            mask[i, :] = 1

    return mask


def main(args):
    torch.set_grad_enabled(False)
    torch.set_float32_matmul_precision("high")
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = True

    diffusion, helper, cfg = utils.inference.setup_model(args.checkpoint)

    accelerator = Accelerator(
        mixed_precision=cfg.mixed_precision,
        dynamo_backend=cfg.dynamo_backend,
        split_batches=True,
        even_batches=False,
        step_scheduler_with_optimizer=True,
    )
    device = accelerator.device

    if accelerator.is_local_main_process:
        print(f"{cfg=}")

    ########################### Different dataloaders for the different datasets ###########################
    # if cfg.dataset == "all":
    #     dataset1 = ds.load_dataset(
    #         path=f"data/synlidar",
    #         name=cfg.lidar_projection,
    #         split=ds.Split.TEST,
    #         num_proc=cfg.num_workers,
    #     ).with_format("torch")

    #     dataset2 = ds.load_dataset(
    #         path=f"data/kitti_raw",
    #         name=cfg.lidar_projection,
    #         split=ds.Split.TEST,
    #         num_proc=cfg.num_workers,
    #     ).with_format("torch")

    #     dataset3 = ds.load_dataset(
    #         path=f"data/kitti_360",
    #         name=cfg.lidar_projection,
    #         split=ds.Split.TEST,
    #         num_proc=cfg.num_workers,
    #     ).with_format("torch")

    #     dataset = ds.concatenate_datasets([dataset1, dataset2, dataset3])

    # else:
    #     dataset = ds.load_dataset(
    #         path="data/kitti_360",
    #         name=cfg.lidar_projection,
    #         split=ds.Split.TEST,
    #         num_proc=cfg.num_workers,
    #     ).with_format("torch")

    # dataset = ds.load_dataset(
    #     path=f"data/synlidar",
    #     name=cfg.lidar_projection,
    #     split=ds.Split.TEST,
    #     num_proc=cfg.num_workers,
    # ).with_format("torch")

    # dataset = ds.load_dataset(
    #     path=f"data/kitti_raw",
    #     name=cfg.lidar_projection,
    #     split=ds.Split.TEST,
    #     num_proc=cfg.num_workers,
    # ).with_format("torch")

    dataset = ds.load_dataset(
        path="data/kitti_360",
        name=cfg.lidar_projection,
        split=ds.Split.TEST,
        num_proc=cfg.num_workers,
    ).with_format("torch")

    #########################################################################################################

    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        num_workers=cfg.num_workers,
        drop_last=False,
        shuffle=False,
    )

    diffusion.to(device)
    sample_fn = torch.compile(
        diffusion.conditional_sample
    )  #### Use this for masked data
    helper, dataloader = accelerator.prepare(helper, dataloader)

    def postprocess(sample):
        """
            Postprocesses data after sampling
        """
        sample = helper.denormalize(sample)
        depth, rflct = sample[:, [0]], sample[:, [1]]
        depth = helper.revert_depth(depth)
        xyz = helper.to_xyz(depth)

        return torch.cat([depth, xyz, rflct], dim=1)

    # Samples all data with a upsampling mask for evaluation
    for batch in tqdm(
        dataloader,
        desc="sampling...",
        dynamic_ncols=True,
        disable=not accelerator.is_main_process,
    ):
        indices = batch["sample_id"].long().to(device)
        depth = batch["depth"].float().to(device)
        depth = helper.convert_depth(depth)
        depth = helper.normalize(depth)
        rflct = batch["reflectance"].float().to(device)
        rflct = helper.normalize(rflct)
        mask = batch["mask"].float().to(device)
        targets = torch.cat([depth, rflct], dim=1)

        ### 2x upsampling ###
        # mask = (
        #     torch.stack([torch.tensor(half_loss(element)) for element in targets])
        #     .unsqueeze(1)
        #     .to(device)
        # )

        ### 4x upsampling ###
        mask = (
            torch.stack([torch.tensor(quarter_loss(element)) for element in targets])
            .unsqueeze(1)
            .to(device)
        )

        with torch.cuda.amp.autocast(enabled=True):
            results = sample_fn(
                batch_size=mask.shape[0],
                num_steps=args.num_steps,
                progress=accelerator.is_main_process,
                rng=torch.Generator(device=device).manual_seed(0),
                mode="ddpm",
                mask=mask.float(),
                x_0=targets,
            ).clamp(-1, 1)

        results = postprocess(results)
        targets = postprocess(targets)

        # Saves the results
        for i in range(len(results)):
            torch.save(
                results[i].clone().cpu(),
                f"{args.output_folder}densification_results_resamplesteps_{args.num_steps}/{indices[i]:010d}.pth",
            )

            target_path = (
                f"{args.output_folder}densification_targets/{indices[i]:010d}.pth"
            )

            if not os.path.isfile(target_path):
                torch.save(
                    targets[i].clone().cpu(),
                    target_path,
                )


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--num_steps", type=int, default=32)
    parser.add_argument("--num_resample_steps", type=int, default=10)
    parser.add_argument("--jump_length", type=int, default=1)
    parser.add_argument("--batch_size", type=int, default=32 * 4)
    parser.add_argument("--output_folder", type=str, default="./")
    args = parser.parse_args()

    os.makedirs(
        f"{args.output_folder}densification_results_resamplesteps_{args.num_steps}",
        exist_ok=True,
    )

    os.makedirs(
        f"{args.output_folder}densification_targets",
        exist_ok=True,
    )

    main(args)
