import os
import numpy as np
import matplotlib.pyplot as plt
import csv
from matplotlib import cbook
from matplotlib import cm
from matplotlib.colors import LightSource
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import LinearLocator
import numpy as np


def analyse_K_L_eval(file: str):
    with open(file, "r") as f:
        reader = csv.reader(f)
        data = np.array(list(reader))
    

    k_values = []
    l_values = []
    for l,k in data[1:, :2]:
        if int(l) not in l_values:
            l_values.append(int(l))
        if int(k) not in k_values:
            k_values.append(int(k))


    mean_values = np.zeros((len(l_values), len(k_values)))
    for l,k,mean in data[1:, :3]:
        if mean == "x":
            mean = np.nan
        mean_values[l_values.index(int(l)), k_values.index(int(k))] = float(mean)

    k_values = np.array(k_values)
    l_values = np.array(l_values)
    mean_values = np.array(mean_values)
    

    fig, ax = plt.subplots(subplot_kw={"projection": "3d"})

    mean_values = np.array(mean_values)
    k_values = np.array(k_values)
    l_values = np.array(l_values)

    min_mean = np.nanmin(mean_values)
    min_indices = np.argwhere(mean_values == min_mean)
    min_l_index, min_k_index = min_indices[0]

    min_l_value = l_values[min_l_index]
    min_k_value = k_values[min_k_index]


    X, Y = np.meshgrid(k_values, l_values)

    surf = ax.plot_surface(X, Y, mean_values, cmap=cm.coolwarm,
                        linewidth=0, antialiased=False)

    ax.zaxis.set_major_locator(LinearLocator(10))
    ax.zaxis.set_major_formatter('{x:.02f}')

    ax.scatter(min_k_value, min_l_value, min_mean, c='red', marker='o', s=100)

    # Display the minimum mean value and its corresponding k and l values
    text_offset = 1
    ax.text(min_k_value, min_l_value, min_mean + text_offset, f'Min Mean: {min_mean:.2f}\nK: {min_k_value}\nL: {min_l_value}',
            color='black', fontsize=12, ha='center', va='center')

    ax.set_xlabel('K')
    ax.set_ylabel('L')
    ax.set_zlabel('Mean')


    ax.set_title('Mean values of K and L for different K and L values')


    plt.show()








if __name__ == '__main__':



    analyse_K_L_eval("results/K_L_evaluation_using_SC_method_15_15_0.1_.csv")
    # analyse_K_L_eval("results/K_L_evaluation_using_UC_method_15_25_0.1_.csv")