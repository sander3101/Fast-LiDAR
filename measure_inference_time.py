# %%
import torch


@torch.inference_mode()
def measure():
    """
        Measures the inference time of the baseline R2DM model, and the conditional model
    """
    model_selection = "ours"

    torch.set_grad_enabled(False)
    torch.set_float32_matmul_precision("high")
    torch.backends.cudnn.benchmark = True
    device = "cuda"

    if model_selection == "R2DM":
        ddpm, _, _ = torch.hub.load(
            "kazuto1011/r2dm",
            "pretrained_r2dm",
            config="r2dm-h-kitti360-300k",
            device=device,
        )
    else:
        import utils.inference

        ddpm, _, _ = utils.inference.setup_model(
            "logs/diffusion/kitti_360/spherical-1024/r2dm_only_quarter_upsample_continous/models/diffusion_0000300000.pth",
            device=device,
        )

    ddpm = torch.compile(ddpm)

    if model_selection == "R2DM": # Repaint sampling
        sample_fn = torch.compile(ddpm.repaint)

    else: # Conditional sampling
        sample_fn = torch.compile(ddpm.conditional_sample)

    batch = 1
    n_warmup = 1
    n_measure = 3
    known = torch.randn(batch, 2, 64, 1024, device=device)
    mask = torch.randn(batch, 1, 64, 1024, device=device)
    kwargs = dict(num_steps=32, num_resample_steps=10)

    # Warm up steps to avoid inconsistent performance due to compilation and caching
    print("warm up")
    for _ in range(n_warmup):
        if model_selection == "R2DM":
            sample_fn(known, mask, **kwargs)
        else:
            sample_fn(batch_size=1, num_steps=8, mode="ddpm", mask=mask, x_0=known)

    # Start measuring the inference times
    print("start")
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(n_measure):
        if model_selection == "R2DM":
            sample_fn(known, mask, **kwargs)
        else:
            sample_fn(batch_size=1, num_steps=8, mode="ddpm", mask=mask, x_0=known)

    end.record()
    torch.cuda.synchronize()
    elapsed_time = start.elapsed_time(end)

    print(elapsed_time / n_measure / 1000, "sec")


if __name__ == "__main__":
    measure()
