""" This file contains functions to analyse the results of the experiments."""
import csv
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import LinearLocator


def analyse_k_l_eval(filename: str):
    """ Analyse the results of the K and L evaluation."""
    with open(filename, "r") as file:
        reader = csv.reader(file)
        data = np.array(list(reader))


    k_values = []
    l_values = []
    for l_value, k_value in data[1:, :2]:
        if float(l_value) not in l_values:
            l_values.append(float(l_value))
        if int(k_value) not in k_values:
            k_values.append(int(k_value))


    mean_values = np.zeros((len(l_values), len(k_values)))
    for l_value,k_value,mean in data[1:, :3]:
        if mean == "x":
            mean = np.nan
        mean_values[l_values.index(float(l_value)), k_values.index(int(k_value))] = float(mean)

    k_values = np.array(k_values)
    l_values = np.array(l_values)
    mean_values = np.array(mean_values)


    ax1 = plt.subplots(subplot_kw={"projection": "3d"})[1]

    mean_values = np.array(mean_values)
    k_values = np.array(k_values)
    l_values = np.array(l_values)

    min_mean = np.nanmin(mean_values)
    min_indices = np.argwhere(mean_values == min_mean)
    min_l_index, min_k_index = min_indices[0]

    min_l_value = l_values[min_l_index]
    min_k_value = k_values[min_k_index]

    ax1.zaxis.set_major_locator(LinearLocator(10))
    ax1.zaxis.set_major_formatter('{x:.02f}')

    ax1.scatter(min_k_value, min_l_value, min_mean, c='red', marker='o', s=100)

    # Display the minimum mean value and its corresponding k_value and l_value values
    text_offset = 1
    ax1.text(min_k_value, min_l_value, min_mean + text_offset, f'Min Mean: {min_mean:.2f}\nK: {min_k_value}\nL: {min_l_value}',
            color='black', fontsize=12, ha='center', va='center')

    ax1.set_xlabel('K')
    ax1.set_ylabel('L')
    ax1.set_zlabel('Mean')


    ax1.set_title('Mean values of K and L for different K and L values')

def algorythme_comparison():
    """ Compare the different algorythms."""
    filename = "results/All_Methods_Performances_Evaluation.csv"
    with open(filename, "r") as file:
        reader = csv.reader(file)
        data = np.array(list(reader))

    method_names = data[1:, 0]
    characteristics = ['Mean', 'Standard Deviation', 'Median', 'Fail Rate (%)', 'Duration (ms)']
    values = data[1:, [2, 3, 6, 7, 8]].astype(float).round(2)

    x_values = np.arange(len(characteristics))  # x_values coordinates for the bars

    # Create subplots with bars for each method's characteristics
    fig, ax1 = plt.subplots()

    num_methods = len(method_names)
    bar_width = 0.15
    offsets = np.linspace(-bar_width*1.5, bar_width*1.5, num_methods)

    bars_list = []  # To store the bars for each method

    for i, method_name in enumerate(method_names):
        bars = ax1.bar(x_values + offsets[i], values[i], bar_width, label=method_name)
        bars_list.append(bars)

    ax1.set_xlabel('Characteristics')
    ax1.set_ylabel('Values')
    ax1.set_title('Method Comparison')
    ax1.set_xticks(x_values)
    ax1.set_xticklabels(characteristics)
    ax1.legend()

    # Manually add labels for each bar
    for bars in bars_list:
        for single_bar in bars:
            height = single_bar.get_height()
            ax1.annotate('{}'.format(height),
                        xy=(single_bar.get_x() + single_bar.get_width() / 2, height),
                        xytext=(0, 3),  # 3 points vertical offset
                        textcoords="offset points",
                        ha='center', va='bottom')

    fig.tight_layout()



def plot_attack_failures_results(files: list[str]):
    """ Plot the attack failures results."""
    if files == []:
        return
    plt.figure()
    for filename in files:
        with open(filename, "r") as file:
            reader = csv.reader(file)
            data = np.array(list(reader))

        number_of_fake_aps = list(range(1, 11))
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
    """ Plot the results of the attack scenarios."""
    if files == []:
        return
    plt.figure()
    success_rates = []
    mean_errors_normal = []
    mean_errors_actual = []
    for filename in files:
        success_rates_file = []
        mean_errors_normal_file = []
        mean_errors_actual_file = []
        with open(filename, "r") as file:
            reader = csv.reader(file)
            data = np.array(list(reader))
        for line in data[1:]:
            success_rates_file.append(float(line[1])*100/(float(line[1])+float(line[2])))
            mean_errors_normal_file.append(float(line[5]))
            mean_errors_actual_file.append(float(line[6]))
        success_rates.append(success_rates_file)
        mean_errors_normal.append(mean_errors_normal_file)
        mean_errors_actual.append(mean_errors_actual_file)
    labels = []
    for filename in files:
        labels.append(" ".join([filename.split("/")[2].split("_")[0]] + filename.split("_")[8:]))
    title = filename.split("_")[6]
    x_values = list(range(1, 11))
    if parameter == "success":
        for i in range(len(files)):
            plt.plot(x_values, success_rates[i], label=labels[i])
        plt.title(title + " : Attack Success Rate")
        plt.ylabel("Success Rate")
    elif parameter == "error_normal":
        for i in range(len(files)):
            plt.plot(x_values, mean_errors_normal[i], label=labels[i])
        plt.title(title + " : Mean Error Normal")
        plt.ylabel("Mean Error")
    elif parameter == "error_actual":
        for i in range(len(files)):
            plt.plot(x_values, mean_errors_actual[i], label=labels[i])
        plt.title(title + " : Mean Error Actual")
        plt.ylabel("Mean Error Actual")
    plt.xlabel("Number of fake APs")
    plt.legend()

def plot_method(files: list[str], parameter="success"):
    """ Plot the results of the attack scenarios."""
    if files == []:
        return
    plt.figure()
    success_rates = []
    mean_errors_normal = []
    mean_errors_actual = []
    for filename in files:
        success_rates_file = []
        mean_errors_normal_file = []
        mean_errors_actual_file = []
        with open(filename, "r") as file:
            reader = csv.reader(file)
            data = np.array(list(reader))
        for line in data[1:]:
            success_rates_file.append(float(line[1])*100/(float(line[1])+float(line[2])))
            mean_errors_normal_file.append(float(line[5]))
            mean_errors_actual_file.append(float(line[6]))
        success_rates.append(success_rates_file)
        mean_errors_normal.append(mean_errors_normal_file)
        mean_errors_actual.append(mean_errors_actual_file)
    title = " ".join(files[0].split("_")[8:])
    labels = []
    for filename in files:
        labels.append(" ".join(filename.split("_")[6:7]))
    x_values = list(range(1, 11))
    if parameter == "success":
        for i in range(len(files)):
            plt.plot(x_values, success_rates[i], label=labels[i])
        plt.title("Attack Success Rate "+ title)
        plt.ylabel("Success Rate")
    elif parameter == "error_normal":
        for i in range(len(files)):
            plt.plot(x_values, mean_errors_normal[i], label=labels[i])
        plt.title("Mean Error"+ title)
        plt.ylabel("Mean Error")
    elif parameter == "error_actual":
        for i in range(len(files)):
            plt.plot(x_values, mean_errors_actual[i], label=labels[i])
        plt.title("Mean Error Actual"+ title)
        plt.ylabel("Mean Error Actual")
    plt.xlabel("Number of fake APs")
    plt.legend()



def analyse_attack_scenarios(comparison: str, files: list[str]):
    """ Analyse the results of the attack scenarios."""

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
        files_method_sc = [file for file in files if "SC" in file]
        files_method_uc = [file for file in files if "UC" in file]
        files_method_vt = [file for file in files if "VT" in file]
        plot_method(files_method_sc, parameter="success")
        plot_method(files_method_uc, parameter="success")
        plot_method(files_method_vt, parameter="success")
        plot_method(files_method_sc, parameter="error_normal")
        plot_method(files_method_uc, parameter="error_normal")
        plot_method(files_method_vt, parameter="error_normal")
        plot_method(files_method_sc, parameter="error_actual")
        plot_method(files_method_uc, parameter="error_actual")
        plot_method(files_method_vt, parameter="error_actual")
        plot_attack_failures_results(files_method_sc)
        plot_attack_failures_results(files_method_uc)
        plot_attack_failures_results(files_method_vt)





if __name__ == '__main__':

    # analyse_k_l_eval("results/K_L_determination/K_L_evaluation_using_SC_method_15_15_0.1_.csv")
    # analyse_k_l_eval("results/K_L_determination/K_L_evaluation_using_UC_method_15_25_0.1_.csv")
    # analyse_k_l_eval("results/K_L_determination/K_L_evaluation_using_VT_method_15_0.01_0.1_.csv")


    basic_files = ["results/corrupted_datasets/basic_knn_on_corrupted_dataset_scenario1_using_SC_method.csv",
             "results/corrupted_datasets/basic_knn_on_corrupted_dataset_scenario2_using_SC_method.csv",
             "results/corrupted_datasets/basic_knn_on_corrupted_dataset_scenario1_using_UC_method.csv",
             "results/corrupted_datasets/basic_knn_on_corrupted_dataset_scenario2_using_UC_method.csv",
             "results/corrupted_datasets/basic_knn_on_corrupted_dataset_scenario1_using_VT_method.csv",
             "results/corrupted_datasets/basic_knn_on_corrupted_dataset_scenario2_using_VT_method.csv",]

    secure_files = ["results/corrupted_datasets/secure_knn_on_corrupted_dataset_scenario1_using_UC_method.csv",
                    "results/corrupted_datasets/secure_knn_on_corrupted_dataset_scenario2_using_UC_method.csv"]

    all_files = basic_files + secure_files

    # analyse_attack_scenarios("method", basic_files)
    # analyse_attack_scenarios("scenario", basic_files)

    # analyse_attack_scenarios("method", secure_files)
    # analyse_attack_scenarios("scenario", secure_files)

    analyse_attack_scenarios("method", all_files)

    # algorythme_comparison()

    plt.show()
