# %%
import matplotlib
import matplotlib.pyplot as plt
from matplotlib import ticker
import json

matplotlib.rcParams["pdf.fonttype"] = 42
matplotlib.rcParams["ps.fonttype"] = 42


def plot_result():
    """
        Plots all metrics in a grid for a given model
    """

    timesteps = [2, 4, 8, 16, 32, 64, 128]
    baselines = [34.44, 0.923, 3.911, 0.050, 0.105]
    results = [[] for i in range(5)]

    for t in timesteps:
        model = json.load(
            open(
                f"sampling_results2/quarter_sampling/only_quarter_upsample/upsampling_scores_resamplesteps_{t}.json",
                "r",
            )
        )

        results[0].append(model["iou"])
        results[1].append(model["MAE-d"])
        results[2].append(model["RMSE-d"])
        results[3].append(model["MAE-r"])
        results[4].append(model["RMSE-r"])

    fig, axs = plt.subplots(1, 5, figsize=(18, 18), constrained_layout=True)
    # fig.delaxes(axs[2, 1])

    # Plot data on each subplot
    axs[0].plot(timesteps, results[0], color="blue", marker="o")
    axs[0].set_title("iou", fontsize=20)

    axs[1].plot(timesteps, results[1], color="blue", marker="o")
    axs[1].set_title("MAE-d", fontsize=20)
    axs[1].set_yscale("log", base=10)

    axs[2].plot(timesteps, results[2], color="blue", marker="o")
    axs[2].set_title("RMSE-d", fontsize=20)
    axs[2].set_yscale("log", base=10)

    axs[3].plot(timesteps, results[3], color="blue", marker="o")
    axs[3].set_title("MAE-r", fontsize=20)
    axs[3].set_yscale("log", base=10)

    axs[4].plot(
        timesteps, results[4], color="blue", marker="o", label="Ours (config C)"
    )
    axs[4].set_title("RMSE-r", fontsize=20)
    axs[4].set_yscale("log", base=10)

    for a, b in zip(axs.flatten(), baselines):
        timesteps = [2, 4, 8, 16, 32, 64, 128, 256, 512]
        a.axhline(y=b, color="black", linestyle="--")
        a.plot(320, b, color="black", linestyle="--", marker="o", label="R2DM")
        a.set_xlabel("Timesteps T", fontsize=20)
        a.set_xscale("log", base=2)
        a.grid(True)
        a.xaxis.set_major_locator(ticker.FixedLocator(timesteps))
        a.xaxis.set_major_formatter(ticker.FixedFormatter(timesteps))
        # a.set_box_aspect(1 / 1.618)
        a.set_box_aspect(1)

    axs[4].legend(loc="upper right", title_fontsize="large", fontsize=15)

    plt.savefig("plot_samples.pdf", bbox_inches="tight", pad_inches=0, dpi=500)
    plt.show()


def plot_result_changed():
    """
        Plots all metrics in a grid for a given model. This function plots the metrics in a more readable plot. The grid is 3x2
    """

    timesteps = [2, 4, 8, 16, 32, 64, 128]
    baselines = [34.44, 0.923, 3.911, 0.050, 0.105]
    results = [[] for i in range(5)]

    for t in timesteps:
        model = json.load(
            open(
                f"sampling_results2/quarter_sampling/only_quarter_upsample/upsampling_scores_resamplesteps_{t}.json",
                "r",
            )
        )
        # model = json.load(
        #     open(
        #         f"sampling_results2/quarter_sampling/all_features_mixed_no_complex/upsampling_scores_resamplesteps_{t}.json",
        #         "r",
        #     )
        # )

        results[0].append(model["iou"])
        results[1].append(model["MAE-d"])
        results[2].append(model["RMSE-d"])
        results[3].append(model["MAE-r"])
        results[4].append(model["RMSE-r"])

    fig, axs = plt.subplots(3, 2, figsize=(12, 20), constrained_layout=True)
    fig.delaxes(axs[2, 1])

    # Plot data on each subplot
    axs[0][0].plot(timesteps, results[0], color="blue", marker="o")
    axs[0][0].set_title("iou", fontsize=20)

    axs[0][1].plot(timesteps, results[1], color="blue", marker="o", label="Config C")
    axs[0][1].set_title("MAE-d", fontsize=20)
    axs[0][1].set_yscale("log", base=10)

    axs[1][0].plot(timesteps, results[2], color="blue", marker="o")
    axs[1][0].set_title("RMSE-d", fontsize=20)
    axs[1][0].set_yscale("log", base=10)

    axs[1][1].plot(timesteps, results[3], color="blue", marker="o")
    axs[1][1].set_title("MAE-r", fontsize=20)
    axs[1][1].set_yscale("log", base=10)

    axs[2][0].plot(
        timesteps, results[4], color="blue", marker="o", label="Ours (config C)"
    )
    axs[2][0].set_title("RMSE-r", fontsize=20)
    axs[2][0].set_yscale("log", base=10)

    for a, b in zip(axs.flatten(), baselines):
        timesteps = [2, 4, 8, 16, 32, 64, 128, 256, 512]
        a.axhline(y=b, color="black", linestyle="--")
        a.plot(320, b, color="black", linestyle="--", marker="o", label="R2DM")
        a.set_xlabel("Timesteps T", fontsize=20)
        a.set_xscale("log", base=2)
        a.grid(True)
        a.xaxis.set_major_locator(ticker.FixedLocator(timesteps))
        a.xaxis.set_major_formatter(ticker.FixedFormatter(timesteps))
        # a.set_box_aspect(1 / 1.618)
        a.set_box_aspect(1)
        a.tick_params(axis="both", which="both", labelsize=15)

    axs[0][1].legend(loc="upper right", title_fontsize="large", fontsize=20)

    plt.savefig("plot_samples.pdf", bbox_inches="tight", pad_inches=0, dpi=500)
    plt.show()


def main():
    # plot_result()
    plot_result_changed()


if __name__ == "__main__":
    main()

# %%
