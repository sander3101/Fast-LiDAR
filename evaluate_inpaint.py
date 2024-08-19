# %%

import argparse
import json
import random
from collections import defaultdict
from pathlib import Path

import einops
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np
import torch
import torch.nn.functional as F
from matplotlib.colors import ListedColormap
from rich import print
from torch.nn.parallel import DataParallel
from tqdm.auto import tqdm

import os

# import render_utils
from metrics import rangenet

# from utils import colorize

torch.set_grad_enabled(False)


def make_semantickitti_cmap():
    """
        Creates colormap for rendering
    """
    label_colors = {
        0: [0, 0, 0],
        1: [245, 150, 100],
        2: [245, 230, 100],
        3: [150, 60, 30],
        4: [180, 30, 80],
        5: [255, 0, 0],
        6: [30, 30, 255],
        7: [200, 40, 255],
        8: [90, 30, 150],
        9: [255, 0, 255],
        10: [255, 150, 255],
        11: [75, 0, 75],
        12: [75, 0, 175],
        13: [0, 200, 255],
        14: [50, 120, 255],
        15: [0, 175, 0],
        16: [0, 60, 135],
        17: [80, 240, 150],
        18: [150, 240, 255],
        19: [0, 0, 255],
    }
    num_classes = max(label_colors.keys()) + 1
    label_colormap = np.zeros((num_classes, 3), dtype=np.uint8)
    for label_id, color in label_colors.items():
        label_colormap[label_id] = color[::-1]  # BGR -> RGB
    cmap = ListedColormap(label_colormap / 255.0)
    return cmap


@torch.no_grad()
def colorize(tensor, cmap_fn=cm.turbo):
    """
        Applies colormap to input tensor
    """
    colors = cmap_fn(np.linspace(0, 1, 256))[:, :3]
    colors = torch.from_numpy(colors).to(tensor)
    tensor = tensor.squeeze(1) if tensor.ndim == 4 else tensor
    ids = (tensor * 256).clamp(0, 255).long()
    tensor = F.embedding(ids, colors).permute(0, 3, 1, 2)
    tensor = tensor.mul(255).clamp(0, 255).byte()
    return tensor


@torch.no_grad()
def evaluate(label, pred, num_classes, epsilon=1e-12):
    """
        Calculates evaluation metrics for sampled results
    """

    # PyTorch version of https://github.com/xuanyuzhou98/SqueezeSegV2/blob/master/src/utils/util.py

    device = label.device
    ious = torch.zeros(num_classes, device=device)
    tps = torch.zeros(num_classes, device=device)
    fns = torch.zeros(num_classes, device=device)
    fps = torch.zeros(num_classes, device=device)
    freqs = torch.zeros(num_classes, device=device)

    for cls_id in range(num_classes):
        tp = (pred[label == cls_id] == cls_id).sum()
        fp = (label[pred == cls_id] != cls_id).sum()
        fn = (pred[label == cls_id] != cls_id).sum()

        ious[cls_id] = tp / (tp + fn + fp + epsilon)
        tps[cls_id] = tp
        fps[cls_id] = fp
        fns[cls_id] = fn
        freqs[cls_id] = (label == cls_id).sum()

    return ious, tps, fps, fns, freqs


class Samples(torch.utils.data.Dataset):
    """
        Dataclass for handling the sampled results from the model
    """

    def __init__(self, result_dir):
        self.result_paths = list(sorted(list(Path(result_dir).glob("*.pth"))))
        print(f"found {len(self.result_paths)} samples")

    def __getitem__(self, index):
        rst_path = self.result_paths[index]
        segment = str(rst_path).split("/")[3]
        target_path = str(rst_path).replace(segment, "densification_targets")
        result = torch.load(rst_path, map_location="cpu")
        gt = torch.load(target_path, map_location="cpu")
        return result, gt

    def __len__(self):
        return len(self.result_paths)


@torch.no_grad()
def main(args):

    device = "cuda"

    print("setting up model...")
    cmap = make_semantickitti_cmap()
    model, preprocess = rangenet.rangenet53(
        weights="SemanticKITTI_64x1024",
        compile=False,
        device=device,
    )

    num_classes = model.num_classes
    model = DataParallel(model)

    loader = torch.utils.data.DataLoader(
        Samples(args.result_dir),
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        drop_last=False,
        shuffle=False,
    )

    mae = lambda batch: batch.abs().mean(dim=[1, 2, 3])
    rmse = lambda batch: batch.square().mean(dim=[1, 2, 3]).sqrt()

    num_samples = 0
    scores = defaultdict(float)
    confusion = defaultdict(int)

    # Run evaluation over all samples
    for pred, gt in tqdm(loader, total=len(loader), desc="evaluating..."):
        pred = pred.to(device)
        gt = gt.to(device)

        diff = pred - gt

        # depth
        scores["MAE-d"] += mae(diff[:, [0]]).sum()
        scores["RMSE-d"] += rmse(diff[:, [0]]).sum()

        # reflectance
        scores["MAE-r"] += mae(diff[:, [4]]).sum()
        scores["RMSE-r"] += rmse(diff[:, [4]]).sum()

        pred_mask = (pred[:, [0]] > 1e-6).float()
        pred_logits = model(preprocess(pred, pred_mask))
        pred_labels = pred_logits.argmax(dim=1) * pred_mask

        gt_mask = (gt[:, [0]] > 1e-6).float()
        gt_logits = model(preprocess(gt, gt_mask))
        gt_labels = gt_logits.argmax(dim=1) * gt_mask

        num_samples += len(pred)

        _, tps, fps, fns, freqs = evaluate(gt_labels, pred_labels, num_classes)
        confusion["tp"] += tps
        confusion["fp"] += fps
        confusion["fn"] += fns
        confusion["freq"] += freqs

    for key in scores:
        scores[key] /= num_samples

    union = confusion["tp"] + confusion["fn"] + confusion["fp"]
    iou = confusion["tp"] / (union + 1e-12)
    scores["iou"] = iou.mean() * 100

    for key in scores:
        scores[key] = float(scores[key])
        print(f"{key:<10}: {scores[key]:.3f}")

    path_parts = str(args.result_dir).split("_")
    timestep = path_parts[len(path_parts) - 1]

    with open(
        args.result_dir.parent / f"upsampling_scores_resamplesteps_{timestep}.json", "w"
    ) as f:
        json.dump(scores, f, indent=4)



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--result_dir", type=Path, required=True)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--num_workers", type=int, default=8)
    args = parser.parse_args()

    main(args)
