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
    mean_errors_normal = []
    mean_errors_actual = []
    for line in data[1:]:
        n_attack_successfull = float(line[1])
        n_attack_failed = float(line[2])
        positioning_failed = float(line[3])
        normal_positioning_failed = float(line[4])
        distance_error_normal_rss = float(line[5])
        distance_error_actual_position = float(line[6])
        total_of_attack = float(line[7])
        success_rates.append(n_attack_successfull*100/(total_of_attack))
        mean_errors_normal.append(distance_error_normal_rss)
        mean_errors_actual.append(distance_error_actual_position)

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
    if files == []:
        return
    plt.figure()
    for file in files:
        with open(file, "r") as f:
            reader = csv.reader(f)
            data = np.array(list(reader))
        
        number_of_fake_aps = [i for i in range(1, 11)]
        fails_rate_due_to_attacks = []
        for line in data[1:]:
            fails_rate_due_to_attacks.append(float(line[3]))
        label = " ".join(file.split("_")[5:8])
        plt.plot(number_of_fake_aps, fails_rate_due_to_attacks, label=label, linestyle='dashed')
    plt.title("Attack Failures")
    plt.xlabel("Number of fake APs")
    plt.ylabel("Success Rate")
    plt.legend()

def plot_scenario(files: list[str], parameter="success"):
    if files == []:
        return
    plt.figure()
    success_rates = []
    mean_errors_normal = []
    mean_errors_actual = []
    for file in files:
        success_rates_file = []
        mean_errors_normal_file = []
        mean_errors_actual_file = []
        with open(file, "r") as f:
            reader = csv.reader(f)
            data = np.array(list(reader))
        for line in data[1:]:
            success_rates_file.append(float(line[1])*100/float(line[7]))
            mean_errors_normal_file.append(float(line[5]))
            mean_errors_actual_file.append(float(line[6]))
        success_rates.append(success_rates_file)
        mean_errors_normal.append(mean_errors_normal_file)
        mean_errors_actual.append(mean_errors_actual_file)
    if parameter == "success":
        for i in range(len(files)):
            plt.plot(success_rates[i], label=" ".join(files[i].split("_")[7:9]))
        plt.title(files[i].split("_")[5] + " : Attack Success Rate")
        plt.ylabel("Success Rate")
    elif parameter == "error_normal":
        for i in range(len(files)):
            plt.plot(mean_errors_normal[i], label=" ".join(files[i].split("_")[7:9]))
        plt.title(files[i].split("_")[5] + " : Mean Error Normal")
        plt.ylabel("Mean Error")
    elif parameter == "error_actual":
        for i in range(len(files)):
            plt.plot(mean_errors_actual[i], label=" ".join(files[i].split("_")[7:9]))
        plt.title(files[i].split("_")[5] + " : Mean Error Actual")
        plt.ylabel("Mean Error Actual")
    plt.xlabel("Number of fake APs")
    plt.legend()

def plot_method(files: list[str], parameter="success"):
    if files == []:
        return
    plt.figure()
    success_rates = []
    mean_errors_normal = []
    mean_errors_actual = []
    for file in files:
        success_rates_file = []
        mean_errors_normal_file = []
        mean_errors_actual_file = []
        with open(file, "r") as f:
            reader = csv.reader(f)
            data = np.array(list(reader))
        for line in data[1:]:
            success_rates_file.append(float(line[1])*100/float(line[7]))
            mean_errors_normal_file.append(float(line[5]))
            mean_errors_actual_file.append(float(line[6]))
        success_rates.append(success_rates_file)
        mean_errors_normal.append(mean_errors_normal_file)
        mean_errors_actual.append(mean_errors_actual_file)
    if parameter == "success":
        for i in range(len(files)):
            plt.plot(success_rates[i], label=" ".join(files[i].split("_")[5:6]))
        plt.title("Attack Success Rate "+ " ".join(files[0].split("_")[7:9]))
        plt.ylabel("Success Rate")
    elif parameter == "error_normal":
        for i in range(len(files)):
            plt.plot(mean_errors_normal[i], label=" ".join(files[i].split("_")[5:6]))
        plt.title("Mean Error"+ " ".join(files[0].split("_")[7:9]))
        plt.ylabel("Mean Error")
    elif parameter == "error_actual":
        for i in range(len(files)):
            plt.plot(mean_errors_actual[i], label=" ".join(files[i].split("_")[5:6]))
        plt.title("Mean Error Actual"+ " ".join(files[0].split("_")[7:9]))
        plt.ylabel("Mean Error Actual")
    plt.xlabel("Number of fake APs")
    plt.legend()
    

    
def analyse_attack_scenarios(comparison: str, files: list[str]):
    
    if comparison == "method":
        files_scenario1 = [file for file in files if "scenario1" in file]
        files_scenario2 = [file for file in files if "scenario2" in file]
        plot_scenario(files_scenario1, parameter="success") 
        plot_scenario(files_scenario2, parameter="success")
        plot_scenario(files_scenario1, parameter="error_normal")
        plot_scenario(files_scenario2, parameter="error_normal")
        plot_scenario(files_scenario1, parameter="error_actual")
        plot_scenario(files_scenario2, parameter="error_actual")
        plot_attack_failures_results(files_scenario1)
        plot_attack_failures_results(files_scenario2)

    elif comparison == "scenario":
        files_method_SC = [file for file in files if "SC" in file]
        files_method_UC = [file for file in files if "UC" in file]
        files_method_VT = [file for file in files if "VT" in file]
        plot_method(files_method_SC, parameter="success")
        plot_method(files_method_UC, parameter="success")
        plot_method(files_method_VT, parameter="success")
        plot_method(files_method_SC, parameter="error_normal")
        plot_method(files_method_UC, parameter="error_normal")
        plot_method(files_method_VT, parameter="error_normal")
        plot_method(files_method_SC, parameter="error_actual")
        plot_method(files_method_UC, parameter="error_actual")
        plot_method(files_method_VT, parameter="error_actual")
        plot_attack_failures_results(files_method_SC)
        plot_attack_failures_results(files_method_UC)
        plot_attack_failures_results(files_method_VT)





if __name__ == '__main__':

    # analyse_K_L_eval("results/K_L_evaluation_using_SC_method_15_15_0.1_.csv")
    # analyse_K_L_eval("results/K_L_evaluation_using_UC_method_15_25_0.1_.csv")
    # analyse_K_L_eval("results/K_L_evaluation_using_VT_method_15_0.05_0.1_.csv")
    # analyse_K_L_eval("results/K_L_evaluation_using_VT_method_15_0.01_0.1_.csv")


    basic_files = ["results/basic_knn_on_corrupted_dataset_scenario1_using_SC_method_K11_L7_.csv",
             "results/basic_knn_on_corrupted_dataset_scenario2_using_SC_method_K11_L7_.csv",
             "results/basic_knn_on_corrupted_dataset_scenario1_using_UC_method_K7_L16_.csv",
             "results/basic_knn_on_corrupted_dataset_scenario2_using_UC_method_K7_L16_.csv",
             "results/basic_knn_on_corrupted_dataset_scenario1_using_VT_method_K4_L0.6_.csv",
             "results/basic_knn_on_corrupted_dataset_scenario2_using_VT_method_K4_L0.6_.csv",]
    
    secure_files = ["results/secure_knn_on_corrupted_dataset_scenario1_using_UC_method_K7_L16_.csv",
                    "results/secure_knn_on_corrupted_dataset_scenario2_using_UC_method_K7_L16_.csv"]

    # analyse_attack_scenarios("method", basic_files)
    # analyse_attack_scenarios("scenario", basic_files)

    # analyse_attack_scenarios("method", secure_files)   
    analyse_attack_scenarios("scenario", secure_files) 

    plt.show()