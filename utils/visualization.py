import matplotlib.pyplot as plt
import numpy as np


def visualize(data):
    data = np.reshape(data, (len(data), 28, 28))
    fig, ax = plt.subplots(10, 10)
    k = 0
    for i in range(10):
        for j in range(10):
            ax[i][j].imshow(data[k], aspect='auto')
            k += 1
    plt.show()