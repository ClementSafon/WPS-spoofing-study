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
        if float(l) not in l_values:
            l_values.append(float(l))
        if int(k) not in k_values:
            k_values.append(int(k))


    mean_values = np.zeros((len(l_values), len(k_values)))
    for l,k,mean in data[1:, :3]:
        if mean == "x":
            mean = np.nan
        mean_values[l_values.index(float(l)), k_values.index(int(k))] = float(mean)

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


    # plt.show()


def plot_attack_success_results(file: str):
    with open(file, "r") as f:
        reader = csv.reader(f)
        data = np.array(list(reader))
    
    number_of_fake_aps = [i for i in range(1, 11)]
    success_rates = []
    mean_errors = []
    for line in data[1:]:
        success_rates.append(float(line[1])*100/(float(line[2])+float(line[1])+float(line[3])))
        mean_errors.append(float(line[5]))

    fig = plt.figure()
    ax1 = fig.add_subplot(111)
    ax1.plot(number_of_fake_aps, success_rates, label="Success Rate", color="blue")
    ax2 = ax1.twinx()
    ax2.plot(number_of_fake_aps, mean_errors, label="Mean Error", color="red")
    title = " ".join([file.split("_")[0].split("/")[1]] + file.split("_")[5:8])
    plt.title(title)
    ax1.set_xlabel("Number of fake APs")
    ax1.set_ylabel("Success Rate")
    ax2.set_ylabel("Mean Error")
    ax1.legend(loc=2)
    ax2.legend(loc=1)

def plot_attack_failures_results(files: list[str]):
    plt.figure()
    for file in files:
        with open(file, "r") as f:
            reader = csv.reader(f)
            data = np.array(list(reader))
        
        number_of_fake_aps = [i for i in range(1, 11)]
        fails_rate_due_to_attacks = []
        for line in data[1:]:
            fails_rate_due_to_attacks.append(float(line[3])/(float(line[2])+float(line[1])))
        label = " ".join(file.split("_")[5:8])
        plt.plot(number_of_fake_aps, fails_rate_due_to_attacks, label=label, linestyle='dashed')
    plt.title("Attack Failures")
    plt.xlabel("Number of fake APs")
    plt.ylabel("Success Rate")
    plt.legend()
    
def analyse_attack_scenarios():
    files = ["results/basic_knn_on_corrupted_dataset_scenario1_using_SC_method_K11_L7_.csv",
             "results/basic_knn_on_corrupted_dataset_scenario2_using_SC_method_K11_L7_.csv",
             "results/basic_knn_on_corrupted_dataset_scenario1_using_OT_method_K7_L16_.csv",
             "results/basic_knn_on_corrupted_dataset_scenario2_using_OT_method_K7_L16_.csv"]
    plot_attack_success_results(files[0])
    plot_attack_success_results(files[1])

    plot_attack_success_results(files[2])
    plot_attack_success_results(files[3])

    plot_attack_failures_results(files)







if __name__ == '__main__':

    # analyse_K_L_eval("results/K_L_evaluation_using_SC_method_15_15_0.1_.csv")
    # analyse_K_L_eval("results/K_L_evaluation_using_UC_method_15_25_0.1_.csv")
    analyse_K_L_eval("results/K_L_evaluation_using_VT_method_15_0.1_0.1_.csv")

    # analyse_attack_scenarios()

    # plot_attack_success_results("results/secure_knn_on_corrupted_dataset_scenario2_using_OT_method_K7_L16_.csv")
    

    plt.show()