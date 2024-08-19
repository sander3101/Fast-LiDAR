# %%
import numpy as np
import json
from matplotlib import pyplot as plt


def read_data(file):
    """
        Reads the loss datafile
    """
    def moving_average(data, n=20):
        """
            Calculates the moving average
        """
        return np.convolve(data, np.ones(n), "same") / n

    f = open(file)

    data = json.load(f)
    data = data

    new_data = [d[1:] for d in data]

    x_values = [nd[0] for nd in new_data]
    y_values = [nd[1] for nd in new_data]
    y_values = moving_average(np.array(y_values))

    plt.ylim(0, 0.1)
    plt.plot(x_values, y_values)


if __name__ == "__main__":
    file = "kitti_360_spherical-1024_r2dm_generate_continuous.json"

    read_data(file)
