""" Main file of the project. Uncomment the function you want to run. """
import time
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.patches import Circle
from tools import graph
import csv
from radio_map import RadioMap
import knn_algorithm as knn
from metadata_gen import load_ap_max

### Simulations ###

## Find the best parameters : SIMU 0X

# Find the best K and LIMIT : SIMU 01
def simu01_sc_method():
    """ Try all the K,LIMIT combinations, for the Shared Coordinate method, and store some results in a csv file."""
    data = [["LIMIT", "K", "MEAN_ERROR", "STD_ERROR", "FAILRATE", "MAX_ERROR", "MIN_ERROR", "MEDIAN_ERROR", "25th_PERCENTILE",
             "75th_PERCENTILE", "90th_PERCENTILE", "95th_PERCENTILE", "99th_PERCENTILE", "99.99th_PERCENTILE"]]

    # Custom Input Data
    size_of_the_sample = len(vld_r_m)
    k_min = 1
    k_max = 15
    limit_max = 15
    limit_min = 1
    tolerance_fail = 0.1

    fgpt_ids = list(range(0, size_of_the_sample))

    for limit in range(limit_max, limit_min - 1, -1):
        for k in range(k_min, k_max + 1):
            position_errors = []
            count_fail = 0
            for i, fgpt_id in enumerate(fgpt_ids):
                print(round((i / size_of_the_sample) * 100,2), " "*(4-len(str(round((i / size_of_the_sample) * 100,0)))) + "%", end="\r")
                position_error = knn.find_position_error(k, limit, trning_r_m, vld_r_m.get_fingerprint(fgpt_id), method="SC")
                if position_error is None:
                    count_fail += 1
                else:
                    position_errors.append(position_error)
                if count_fail / size_of_the_sample > tolerance_fail:
                    position_errors = []
                    break
            if len(position_errors) > 0:
                data.append([limit, k, np.mean(position_errors), np.std(position_errors), count_fail/size_of_the_sample,
                             np.max(position_errors), np.min(position_errors), np.median(position_errors),
                             np.percentile(position_errors, 25), np.percentile(position_errors, 75),
                             np.percentile(position_errors, 90), np.percentile(position_errors, 95),
                             np.percentile(position_errors, 99), np.percentile(position_errors, 99.99)])
                print("K=", k, " LIMIT=", limit, " -> ", round(np.mean(position_errors),2))
            else:
                print("K=", k, " LIMIT=", limit, " -> ", "x                                  ")
                for k in range(k, k_max + 1):
                    data.append([limit, k, "x", "x", "x", "x", "x", "x", "x", "x", "x", "x", "x", "x"])
                break

    csv_file = "results/K_L_evaluation_using_SC_method_"+str(k_max)+"_"+str(limit_max)+"_"+str(tolerance_fail)+"_.csv"

    with open(csv_file, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerows(data)

    print(f"CSV file '{csv_file}' created successfully.")

def simu01_uc_method():
    """ Try all the K,LIMIT combinations, for the Unshared Coordinate method, and store some results in a csv file."""
    data = [["LIMIT", "K", "MEAN_ERROR", "STD_ERROR", "FAILRATE", "MAX_ERROR", "MIN_ERROR", "MEDIAN_ERROR", "25th_PERCENTILE",
             "75th_PERCENTILE", "90th_PERCENTILE", "95th_PERCENTILE", "99th_PERCENTILE", "99.99th_PERCENTILE"]]

    # Custom Input Data
    size_of_the_sample = len(vld_r_m)
    k_min = 1
    k_max = 15
    limit_max = 25
    limit_min = 10
    tolerance_fail = 0.1

    fgpt_ids = list(range(0, size_of_the_sample))

    for limit in range(limit_min, limit_max + 1):
        for k in range(k_min, k_max + 1):
            position_errors = []
            count_fail = 0
            for i, fgpt_id in enumerate(fgpt_ids):
                print(round((i / size_of_the_sample) * 100,2), " "*(4-len(str(round((i / size_of_the_sample) * 100,0)))) + "%", end="\r")
                position_error = knn.find_position_error(k, limit, trning_r_m, vld_r_m.get_fingerprint(fgpt_id), method="UC")
                if position_error is None:
                    count_fail += 1
                else:
                    position_errors.append(position_error)
                if count_fail / size_of_the_sample > tolerance_fail:
                    position_errors = []
                    break
            if len(position_errors) > 0:
                data.append([limit, k, np.mean(position_errors), np.std(position_errors), count_fail/size_of_the_sample,
                             np.max(position_errors), np.min(position_errors), np.median(position_errors),
                             np.percentile(position_errors, 25), np.percentile(position_errors, 75),
                             np.percentile(position_errors, 90), np.percentile(position_errors, 95),
                             np.percentile(position_errors, 99), np.percentile(position_errors, 99.99)])
                print("K=", k, " LIMIT=", limit, " -> ", round(np.mean(position_errors),2))
            else:
                print("K=", k, " LIMIT=", limit, " -> ", "x                                  ")
                for k in range(k, k_max + 1):
                    data.append([limit, k, "x", "x", "x", "x", "x", "x", "x", "x", "x", "x", "x", "x"])
                break

    csv_file = "results/K_L_evaluation_using_UC_method_"+str(k_max)+"_"+str(limit_max)+"_"+str(tolerance_fail)+"_.csv"

    with open(csv_file, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerows(data)

    print(f"CSV file '{csv_file}' created successfully.")

def simu01_vt_method():
    """ Try all the K,LIMIT combinations, for the Variable Threshold method, and store some results in a csv file."""
    data = [["LIMIT", "K", "MEAN_ERROR", "STD_ERROR", "FAILRATE", "MAX_ERROR", "MIN_ERROR", "MEDIAN_ERROR", "25th_PERCENTILE",
            "75th_PERCENTILE", "90th_PERCENTILE", "95th_PERCENTILE", "99th_PERCENTILE", "99.99th_PERCENTILE"]]

    # Custom Input Data
    size_of_the_sample = len(vld_r_m)
    k_min = 1
    k_max = 15
    limit_max = 0.8
    limit_min = 0.5
    limit_pas = 0.01
    tolerance_fail = 0.1

    fgpt_ids = list(range(0, size_of_the_sample))

    for limit in np.linspace(limit_min, limit_max, int((limit_max - limit_min) / limit_pas) + 1):
        for k in range(k_min, k_max + 1):
            position_errors = []
            count_fail = 0
            for i, fgpt_id in enumerate(fgpt_ids):
                print(round((i / size_of_the_sample) * 100,2), " "*(4-len(str(round((i / size_of_the_sample) * 100,0)))) + "%", end="\r")
                position_error = knn.find_position_error(k, limit, trning_r_m, vld_r_m.get_fingerprint(fgpt_id), method="VT")
                if position_error is None:
                    count_fail += 1
                else:
                    position_errors.append(position_error)
                if count_fail / size_of_the_sample > tolerance_fail:
                    position_errors = []
                    break
            if len(position_errors) > 0:
                data.append([limit, k, np.mean(position_errors), np.std(position_errors), count_fail/size_of_the_sample,
                             np.max(position_errors), np.min(position_errors), np.median(position_errors),
                             np.percentile(position_errors, 25), np.percentile(position_errors, 75),
                             np.percentile(position_errors, 90), np.percentile(position_errors, 95),
                             np.percentile(position_errors, 99), np.percentile(position_errors, 99.99)])
                print("K=", k, " LIMIT=", limit, " -> ", round(np.mean(position_errors),2))
            else:
                print("K=", k, " LIMIT=", limit, " -> ", "x                                  ")
                for k in range(k, k_max + 1):
                    data.append([limit, k, "x", "x", "x", "x", "x", "x", "x", "x", "x", "x", "x", "x"])
                break

    csv_file = "results/K_L_evaluation_using_VT_method_"+str(k_max)+"_"+str(limit_pas)+"_"+str(tolerance_fail)+"_.csv"

    with open(csv_file, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerows(data)

    print(f"CSV file '{csv_file}' created successfully.")


# Test the best K and LIMIT : SIMU 02
def simu02_sc_method():
    """ find the error for a k, and limit combination."""

    k = 11
    limit = 7

    errors = []
    failed = 0
    sum_durations = 0
    for fgpt_id in range(len(vld_r_m)):
        print(round((fgpt_id / len(vld_r_m)) * 100,1), " "*(4-len(str(round((fgpt_id / len(vld_r_m)) * 100,1)))) + "%", end="\r")
        error = knn.find_position_error(k, limit, trning_r_m, vld_r_m.get_fingerprint(fgpt_id), method="SC")
        sum_durations += knn.duration
        if error is not None:
            errors.append(error)
        else:
            failed += 1

    print(f"""
    K={k}
    LIMIT={limit}
    (mean error) {np.mean(errors)}
    (std error) {np.std(errors)}
    (max error) {np.max(errors)}
    (min error) {np.min(errors)}
    (median error) {np.median(errors)}
    """)
    print("Failed: ", failed, "/", len(vld_r_m), " -> ", round(failed*100 / len(vld_r_m),2))
    print("Time to compute one position : ", round((sum_durations/len(vld_r_m))*1000,2), "ms")

def simu02_uc_method():
    """ find the error for a k, and limit combination."""

    k = 7
    limit = 16

    errors = []
    failed = 0
    sum_durations = 0
    for fgpt_id in range(len(vld_r_m)):
        print(round((fgpt_id / len(vld_r_m)) * 100,1), " "*(4-len(str(round((fgpt_id / len(vld_r_m)) * 100,1)))) + "%", end="\r")
        error = knn.find_position_error(k, limit, trning_r_m, vld_r_m.get_fingerprint(fgpt_id), method="UC")
        if error is not None:
            errors.append(error)
        else:
            failed += 1

    print(f"""
    K={k}
    LIMIT={limit}
    (mean error) {np.mean(errors)}
    (std error) {np.std(errors)}
    (max error) {np.max(errors)}
    (min error) {np.min(errors)}
    (median error) {np.median(errors)}
    """)
    print("Failed: ", failed, "/", len(vld_r_m), " -> ", round(failed*100 / len(vld_r_m),2))
    print("Time to compute one position : ", round((sum_durations/len(vld_r_m))*1000,2), "ms")

def simu02_vt_method():
    """ find the error for a k, and limit combination."""

    k = 11
    limit = 0.64

    errors = []
    failed = 0
    sum_durations = 0
    for fgpt_id in range(len(vld_r_m)):
        print(round((fgpt_id / len(vld_r_m)) * 100,1), " "*(4-len(str(round((fgpt_id / len(vld_r_m)) * 100,1)))) + "%", end="\r")
        error = knn.find_position_error(k, limit, trning_r_m, vld_r_m.get_fingerprint(fgpt_id), method="VT")
        if error is not None:
            errors.append(error)
        else:
            failed += 1

    print(f"""
    K={k}
    LIMIT={limit}
    (mean error) {np.mean(errors)}
    (std error) {np.std(errors)}
    (max error) {np.max(errors)}
    (min error) {np.min(errors)}
    (median error) {np.median(errors)}
    """)
    print("Failed: ", failed, "/", len(vld_r_m), " -> ", round(failed*100 / len(vld_r_m),2))
    print("Time to compute one position : ", round((sum_durations/len(vld_r_m))*1000,2), "ms")


# Get performance for each methods with the best parameters : SIMU 03
def simu02_all():
    """ find the performance for each methods. """

    basic_methods = ["SC", "UC", "VT"]
    secure_methods = ["UC,OF,0"]

    data = [["method", "(k,l)", "mean_error", "std_error", "max_error",
             "min_error", "median_error", "failed", "time_to_compute_one_position"]]

    for method in basic_methods:
        errors = []
        failed = 0
        sum_durations = 0
        print("Method: (basic)", method)
        for fgpt_id in range(len(vld_r_m)):
            print(round((fgpt_id / len(vld_r_m)) * 100,1), " "*(4-len(str(round((fgpt_id / len(vld_r_m)) * 100,1)))) + "%", end="\r")
            error = knn.find_position_error(k_l_values[method][0], k_l_values[method][1], trning_r_m, vld_r_m.get_fingerprint(fgpt_id), method=method)
            sum_durations += knn.duration
            if error is not None:
                errors.append(error)
            else:
                failed += 1

        data.append([method + " (basic)", k_l_values[method], np.mean(errors), np.std(errors), np.max(errors),
                     np.min(errors), np.median(errors), round(failed*100/len(vld_r_m),2), round((sum_durations/len(vld_r_m))*1000,2)])

    for method in secure_methods:
        tolerance = float(method.split(",")[2])
        filter_type = method.split(",")[1]
        method = method.split(",", maxsplit=1)[0]
        errors = []
        failed = 0
        sum_durations = 0
        print("Method: (secure)", method)
        for fgpt_id in range(len(vld_r_m)):
            print(round((fgpt_id / len(vld_r_m)) * 100,1), " "*(4-len(str(round((fgpt_id / len(vld_r_m)) * 100,1)))) + "%", end="\r")
            k_l_value = k_l_values[method]
            error = knn.find_position_error(k_l_value[0], k_l_value[1], trning_r_m, vld_r_m.get_fingerprint(fgpt_id), method, filter_type, tolerance)
            sum_durations += knn.duration
            if error is not None:
                errors.append(error)
            else:
                failed += 1

        data.append([method + " (secure)", k_l_values[method], np.mean(errors), np.std(errors), np.max(errors),
                     np.min(errors), np.median(errors), round(failed*100/len(vld_r_m),2), round((sum_durations/len(vld_r_m))*1000,2)])


    csv_file = "results/All_Methods_Performances_Evaluation.csv"

    with open(csv_file, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerows(data)

    print(f"CSV file '{csv_file}' created successfully.")


############################################################################################################


## Test Basic KNN-Algo on Corrupted Data : SIMU 1X

def simu1x_scenariox_xx_method(scenario: str, method: str):
    """ Generic function to find some results for a corrupted validation dataset."""
    data = [["FILE", "ATTACK_SUCCESSFUL", "ATTACK_FAILED", "POSITIONING_FAIL", "NORMAL_POSITIONING_FAIL",
             "MEAN_ERROR_NORMAL_RSS", "MEAN_ERROR_ACTUAL_POSITION", "TOTAL_OF_ATTACK"]]

    k = k_l_values[method][0]
    limit = k_l_values[method][1]

    for file_index in range(1,11):
        vld_x_r_m = RadioMap()
        vld_x_r_m.load_from_csv('datasets/corrupted/' + scenario + '/ValidationData_' + str(file_index) + '.csv')

        n_attack_successfull = 0
        n_attack_failed = 0
        positioning_failed = 0
        normal_positioning_failed = 0
        distance_error_normal_rss = []
        distance_error_actual_position = []
        total_of_attack = len(vld_x_r_m)
        print('ValidationData_' + str(file_index) + '.csv')
        for fgpt_id in range(len(vld_x_r_m)):
            print(round((fgpt_id / len(vld_r_m)) * 100,1), " "*(4-len(str(round((fgpt_id / len(vld_r_m)) * 100,1)))) + "%", end="\r")
            predicted_position = knn.find_position(k, limit, trning_r_m, vld_x_r_m.get_fingerprint(fgpt_id), method)
            normal_predicted_position = knn.find_position(k, limit, trning_r_m, vld_r_m.get_fingerprint(fgpt_id), method)
            actual_position = vld_x_r_m.get_fingerprint(fgpt_id).get_position()
            null_pred_pos = (predicted_position == [0,0,0]).all()
            null_norm_pos = (normal_predicted_position == [0,0,0]).all()

            if null_norm_pos:
                normal_positioning_failed += 1
            if null_pred_pos:
                positioning_failed += 1
                n_attack_failed += 1
            else:
                if (predicted_position != normal_predicted_position).any():
                    n_attack_successfull += 1
                    if not null_norm_pos and not null_pred_pos:
                        distance_error_normal_rss.append(np.linalg.norm(predicted_position - normal_predicted_position))
                        distance_error_actual_position.append(np.linalg.norm(predicted_position - actual_position))
                else:
                    n_attack_failed += 1

        data.append(["ValidationData_" + str(file_index) + '.csv', n_attack_successfull, n_attack_failed, positioning_failed,
                     normal_positioning_failed, np.mean(distance_error_normal_rss), np.mean(distance_error_actual_position), total_of_attack])

    csv_file = "results/corrupted_datasets/basic_knn_on_corrupted_dataset_" + scenario + "_using_" + method + "_method.csv"

    with open(csv_file, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerows(data)
    print(f"CSV file '{csv_file}' created successfully.")

# Scenario 1
def simu11_scenario1_sc_method():
    """ Find some results for a corrupted validation dataset with scenario1."""

    scenario = "scenario1"
    method = "SC"

    simu1x_scenariox_xx_method(scenario, method)

def simu11_scenario1_uc_method():
    """ Find some results for a corrupted validation dataset with scenario1."""

    scenario = "scenario1"
    method = "UC"

    simu1x_scenariox_xx_method(scenario, method)

def simu11_scenario1_vt_method():
    """ Find some results for a corrupted validation dataset with scenario1."""

    scenario = "scenario1"
    method = "VT"

    simu1x_scenariox_xx_method(scenario, method)

# Scenario 2
def simu12_scenario2_sc_method():
    """ Find some results for a corrupted validation dataset with scenario2."""

    scenario = "scenario2"
    method = "SC"

    simu1x_scenariox_xx_method(scenario, method)

def simu12_scenario2_uc_method():
    """ Find some results for a corrupted validation dataset with scenario2."""

    scenario = "scenario2"
    method = "UC"

    simu1x_scenariox_xx_method(scenario, method)

def simu12_scenario2_vt_method():
    """ Find some results for a corrupted validation dataset with scenario2."""

    scenario = "scenario2"
    method = "VT"

    simu1x_scenariox_xx_method(scenario, method)


#############################################################################################################

## Test Secure KNN-Algo on Corrupted Data : SIMU 2X

## No filter
def simu20_single_test():
    """ find the error for a k, and limit combination."""

    n_fake_aps = 8
    scenario = "scenario2"
    method = "VT"

    k = k_l_values[method][0]
    limit = k_l_values[method][1]

    vld_x_r_m = RadioMap()
    vld_x_r_m.load_from_csv('datasets/corrupted/' + scenario + '/ValidationData_' + str(n_fake_aps) + '.csv')

    errors = []
    failed = 0
    for fgpt_id in range(len(vld_x_r_m)):
        print(round((fgpt_id / len(vld_x_r_m)) * 100,1), " "*(4-len(str(round((fgpt_id / len(vld_x_r_m)) * 100,1)))) + "%", end="\r")
        error = knn.find_position_error(k, limit, trning_r_m, vld_x_r_m.get_fingerprint(fgpt_id), method)
        if error is not None:
            errors.append(error)
        else:
            failed += 1

    print(f"""
    K={k}
    LIMIT={limit}
    (mean error) {np.mean(errors)}
    (std error) {np.std(errors)}
    (max error) {np.max(errors)}
    (min error) {np.min(errors)}
    (median error) {np.median(errors)}
    """)
    print("Failed: ", failed, "/", len(vld_x_r_m), " -> ", round(failed*100 / len(vld_x_r_m),2))


## Scenario 1/2
def simu2x_scenariox_xx_method(scenario: str, method: str, filter_type: str, tolerance: float = 0):
    """ Generic function to find some results for a corrupted validation dataset."""
    data = [["FILE", "ATTACK_SUCCESSFUL", "ATTACK_FAILED", "POSITIONING_FAIL", "NORMAL_POSITIONING_FAIL",
             "MEAN_ERROR_NORMAL_RSS", "MEAN_ERROR_ACTUAL_POSITION", "TOTAL_OF_ATTACK"]]

    k = k_l_values[method][0]
    limit = k_l_values[method][1]

    for file_index in range(1,11):
        vld_x_r_m = RadioMap()
        vld_x_r_m.load_from_csv('datasets/corrupted/' + scenario + '/ValidationData_' + str(file_index) + '.csv')

        n_attack_successfull = 0
        n_attack_failed = 0
        positioning_failed = 0
        normal_positioning_failed = 0
        distance_error_normal_rss = []
        distance_error_actual_position = []
        total_of_attack = 0
        print('ValidationData_' + str(file_index) + '.csv')
        for fgpt_id in range(len(vld_x_r_m)):
            print(round((fgpt_id / len(vld_r_m)) * 100,1), " "*(4-len(str(round((fgpt_id / len(vld_r_m)) * 100,1)))) + "%", end="\r")
            predicted_position = knn.find_position(k, limit, trning_r_m, vld_x_r_m.get_fingerprint(fgpt_id), method, filter_type, tolerance)
            normal_predicted_position = knn.find_position(k, limit, trning_r_m, vld_r_m.get_fingerprint(fgpt_id), method, filter_type, tolerance)
            actual_position = vld_x_r_m.get_fingerprint(fgpt_id).get_position()
            null_pred_pos = (predicted_position == [0,0,0]).all()
            null_norm_pos = (normal_predicted_position == [0,0,0]).all()


            if null_norm_pos:
                normal_positioning_failed += 1
            if null_pred_pos:
                positioning_failed += 1
                n_attack_failed += 1
            else:
                if (predicted_position != normal_predicted_position).any():
                    n_attack_successfull += 1
                    if not null_norm_pos and not null_pred_pos:
                        distance_error_normal_rss.append(np.linalg.norm(predicted_position - normal_predicted_position))
                        distance_error_actual_position.append(np.linalg.norm(predicted_position - actual_position))
                else:
                    n_attack_failed += 1


            total_of_attack = len(vld_x_r_m)

        data.append(["ValidationData_" + str(file_index) + '.csv', n_attack_successfull, n_attack_failed, positioning_failed,
                     normal_positioning_failed, np.mean(distance_error_normal_rss), np.mean(distance_error_actual_position), total_of_attack])

    csv_file = "results/corrupted_datasets/secure_knn_on_corrupted_dataset_" + scenario + "_using_" + method + "_method.csv"

    with open(csv_file, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerows(data)
    print(f"CSV file '{csv_file}' created successfully.")

# Overall filter : SIMU 21
def simu21_scenario1_uc_method_secu():
    """ Find some results for a corrupted validation dataset with scenario1 and a secure Overall Filter layer."""

    scenario = "scenario1"
    method = "UC"
    filter_type = "OF"
    tolerance = 0

    simu2x_scenariox_xx_method(scenario, method, filter_type, tolerance)

def simu21_scenario2_uc_method_secu():
    """ Find some results for a corrupted validation dataset with scenario2 and a secure Overall Filter layer."""

    scenario = "scenario2"
    method = "UC"
    filter_type = "OF"
    tolerance = 0

    simu2x_scenariox_xx_method(scenario, method, filter_type, tolerance)


# Precise filter : SIMU 22
def simu22_scenario1_uc_method_secu():
    """ Find some results for a corrupted validation dataset with scenario1 and a secure Precise Filter layer."""

    scenario = "scenario1"
    method = "UC"
    filter_type = "PF"
    tolerance = 0

    simu2x_scenariox_xx_method(scenario, method, filter_type, tolerance)

def simu22_scenario2_uc_method_secu():
    """ Find some results for a corrupted validation dataset with scenario2 and a secure Precise Filter layer."""

    scenario = "scenario2"
    method = "UC"
    filter_type = "PF"
    tolerance = 0

    simu2x_scenariox_xx_method(scenario, method, filter_type, tolerance)



# False Positive Analysis : SIMU 23
def simu23_overall_filter():
    """ Find positioning errors on the validation dataset with a Overall Filter."""

    method = "UC"
    filter_type = "OF"
    tolerance = 0

    k = k_l_values[method][0]
    limit = k_l_values[method][1]

    errors = []
    failed = 0
    for fgpt_id in range(len(vld_r_m)):
        print(round((fgpt_id / len(vld_r_m)) * 100,1), " "*(4-len(str(round((fgpt_id / len(vld_r_m)) * 100,1)))) + "%", end="\r")
        error = knn.find_position_error(k, limit, trning_r_m, vld_r_m.get_fingerprint(fgpt_id), method, filter_type, tolerance)
        if error is not None:
            errors.append(error)
        else:
            failed += 1

    print(f"""
    K={k}
    LIMIT={limit}
    (mean error) {np.mean(errors)}
    (std error) {np.std(errors)}
    (max error) {np.max(errors)}
    (min error) {np.min(errors)}
    (median error) {np.median(errors)}
    """)
    print("Failed: ", failed, "/", len(vld_r_m), " -> ", round(failed*100 / len(vld_r_m),2))

def simu23_precise_filter():
    """ Find positioning errors on the validation dataset with a Precise Filter."""

    method = "UC"
    filter_type = "PF"
    tolerance = 0

    k = k_l_values[method][0]
    limit = k_l_values[method][1]

    errors = []
    failed = 0
    for fgpt_id in range(len(vld_r_m)):
        print(round((fgpt_id / len(vld_r_m)) * 100,1), " "*(4-len(str(round((fgpt_id / len(vld_r_m)) * 100,1)))) + "%", end="\r")
        error = knn.find_position_error(k, limit, trning_r_m, vld_r_m.get_fingerprint(fgpt_id), method, filter_type, tolerance)
        if error is not None:
            errors.append(error)
        else:
            failed += 1

    print(f"""
    K={k}
    LIMIT={limit}
    (mean error) {np.mean(errors)}
    (std error) {np.std(errors)}
    (max error) {np.max(errors)}
    (min error) {np.min(errors)}
    (median error) {np.median(errors)}
    """)
    print("Failed: ", failed, "/", len(vld_r_m), " -> ", round(failed*100 / len(vld_r_m),2))


# Single tests : SIMU 24
def simu24_single_test_of():
    """ find the error for a k, and limit combination."""

    method = "UC"
    filter_type = "OF"
    tolerance = 0
    n_fake_aps = 5
    scenario = "scenario1"
    # fgpt_ids = [i for i in range(0, len(vld_r_m))]
    fgpt_ids = [919]

    k = k_l_values[method][0]
    limit = k_l_values[method][1]

    vld_x_r_m = RadioMap()
    vld_x_r_m.load_from_csv('datasets/corrupted/' + scenario + '/ValidationData_' + str(n_fake_aps) + '.csv')


    errors = []
    failed = 0
    id_error = []
    for fgpt_id in fgpt_ids:
        print(round((fgpt_id / len(vld_x_r_m)) * 100,1), " "*(4-len(str(round((fgpt_id / len(vld_x_r_m)) * 100,1)))) + "%", end="\r")
        error = knn.find_position_error(k, limit, trning_r_m, vld_x_r_m.get_fingerprint(fgpt_id), method, filter_type, tolerance)
        normal_error = knn.find_position_error(k, limit, trning_r_m, vld_r_m.get_fingerprint(fgpt_id), method, filter_type, tolerance)
        print(knn.find_position(k, limit, trning_r_m, vld_x_r_m.get_fingerprint(fgpt_id), method, filter_type, tolerance))
        if error is not None and (np.round(error, 3) != np.round(normal_error, 3)).any():
            errors.append(error)
            id_error.append(fgpt_id)
        else:
            failed += 1

    print(f"""
    K={k}
    LIMIT={limit}
    (mean error) {np.mean(errors)}
    (std error) {np.std(errors)}
    (max error) {np.max(errors)}
    (min error) {np.min(errors)}
    (median error) {np.median(errors)}
    """)
    print("Failed: ", failed, "/", len(vld_x_r_m), " -> ", round(failed*100 / len(vld_x_r_m),2))

    if len(errors) < 15:
        print(errors, id_error)

def simu23_single_test_pf():
    """ find the error for a k, and limit combination."""

    method = "UC"
    filter_type = "PF"
    tolerance = 0
    n_fake_aps = 8
    scenario = "scenario2"

    k = k_l_values[method][0]
    limit = k_l_values[method][1]

    vld_x_r_m = RadioMap()
    vld_x_r_m.load_from_csv('datasets/corrupted/' + scenario + '/ValidationData_' + str(n_fake_aps) + '.csv')


    errors = []
    failed = 0
    for fgpt_id in range(len(vld_x_r_m)):
        print(round((fgpt_id / len(vld_x_r_m)) * 100,1), " "*(4-len(str(round((fgpt_id / len(vld_x_r_m)) * 100,1)))) + "%", end="\r")
        error = knn.find_position_error(k, limit, trning_r_m, vld_x_r_m.get_fingerprint(fgpt_id), method, filter_type, tolerance)
        if error is not None:
            errors.append(error)
        else:
            failed += 1

    print(f"""
    K={k}
    LIMIT={limit}
    (mean error) {np.mean(errors)}
    (std error) {np.std(errors)}
    (max error) {np.max(errors)}
    (min error) {np.min(errors)}
    (median error) {np.median(errors)}
    """)
    print("Failed: ", failed, "/", len(vld_x_r_m), " -> ", round(failed*100 / len(vld_x_r_m),2))


##############################################################################################################

# Other simulations
def display_AP_fingerprints(id_AP: int):
    """ Display the fingerprints of the AP with the id id_AP. """
    x_coords, y_coords = [], []
    rss_values = []
    timestamps = []
    for fingerprint in trning_r_m.get_fingerprints():
        rssi = fingerprint.get_rss()[id_AP - 1]
        if rssi != 100:
            fgpt_lon = fingerprint.get_position()[0]
            fgpt_lat = fingerprint.get_position()[1]
            fgpt_timestamp = fingerprint.get_timestamp()
            if fgpt_lon in x_coords and fgpt_lat == y_coords[x_coords.index(fgpt_lon)]:
                index = x_coords.index(fgpt_lon)
                if fgpt_timestamp > timestamps[index]:
                    rss_values[index] = rssi
                    timestamps[index] = fgpt_timestamp
            else:
                x_coords.append(fgpt_lon)
                y_coords.append(fgpt_lat)
                rss_values.append(rssi)
                timestamps.append(fgpt_timestamp)
    if len(rss_values) == 0:
        print("No data for this AP")
        return

    normalized_timestamps = (np.array(timestamps) - np.min(timestamps)) / (np.max(timestamps) - np.min(timestamps))

    graph.plot_radio_map(trning_r_m, new_figure=True, alpha=0.3)
    colormap = plt.colormaps.get_cmap('viridis')
    scatter = plt.scatter(x_coords, y_coords, c=rss_values, cmap=colormap, marker='o', s=60, vmin=-110, vmax=0)
    cbar = plt.colorbar(scatter)
    cbar.set_label('RSSI')
    plt.xlabel('X Coordinate')
    plt.ylabel('Y Coordinate')
    plt.title('RSSI of AP ' + str(id_AP) + " (normalized)\nn_fgpts = " + str(len(rss_values)))

    graph.plot_radio_map(trning_r_m, new_figure=True, alpha=0.3)
    scatter = plt.scatter(x_coords, y_coords, c=normalized_timestamps, cmap=colormap, marker='o', s=60)
    cbar = plt.colorbar(scatter)
    cbar.set_label('Timestamp')
    plt.xlabel('X Coordinate')
    plt.ylabel('Y Coordinate')
    plt.title('Timestamp of AP ' + str(id_AP) + " (normalized)\nn_fgpts = " + str(len(rss_values)))

    plt.show()

def visualize_ap_centers():
    """ Plot the filtering algorithm principle. """
    fgpt_ids = [919]
    scenario = "scenario2"
    file_index = 8

    vld_x_r_m = RadioMap()
    vld_x_r_m.load_from_csv('datasets/corrupted/' + scenario + '/ValidationData_' + str(file_index) + '.csv')

    for fgpt_id in fgpt_ids:
        fgpt = vld_r_m.get_fingerprint(fgpt_id)
        max_dist, centers = load_ap_max(trning_r_m)
        graph.plot_radio_map(trning_r_m, new_figure=True, alpha=0.3, title="Valid fingerprint : " + str(fgpt_id))
        for i, rss in enumerate(fgpt.get_rss()):
            if rss != 100:
                if centers[i,0] != 0:
                    plt.scatter(centers[i,0], centers[i,1], marker='x', color='red')
                    circle = Circle((centers[i,0], centers[i,1]), max_dist[i] / 2, color='red', fill=False)
                    plt.gca().add_patch(circle)
        plt.scatter(fgpt.get_position()[0], fgpt.get_position()[1], marker='o', color='green')


    for fgpt_id in fgpt_ids:
        fgpt = vld_x_r_m.get_fingerprint(fgpt_id)
        max_dist, centers = load_ap_max(trning_r_m)
        graph.plot_radio_map(trning_r_m, new_figure=True, alpha=0.3, title="Corrupted fingerprint : " + str(fgpt_id))
        for i, rss in enumerate(fgpt.get_rss()):
            if rss != 100:
                if centers[i,0] != 0:
                    plt.scatter(centers[i,0], centers[i,1], marker='x', color='red')
                    circle = Circle((centers[i,0], centers[i,1]), max_dist[i] / 2, color='red', fill=False)
                    plt.gca().add_patch(circle)
        plt.scatter(fgpt.get_position()[0], fgpt.get_position()[1], marker='o', color='green')

    plt.show()
    return

def visualize_ap_centers_vs_fingerprint():
    """ Plot the ap estimated center and the corresponding fingerprints. """
    ap_row = 125
    fgpt_positions = []

    for fgpt in vld_r_m.get_fingerprints():
        if fgpt.get_rss()[ap_row] != 100:
            fgpt_positions.append(fgpt.get_position())
    
    graph.plot_radio_map(trning_r_m, title="Fgpt positions for AP " + str(ap_row), alpha=0.4)
    for position in fgpt_positions:
        plt.scatter(position[0], position[1], marker='o', color='green')
    ap_diameters, estimated_centers = load_ap_max(trning_r_m)
    ap_diameter , estimated_center = ap_diameters[ap_row], estimated_centers[ap_row]
    plt.scatter(estimated_center[0], estimated_center[1], marker='x', color='red')
    circle = Circle((estimated_center[0], estimated_center[1]), ap_diameter / 2, color='red', fill=False)
    plt.gca().add_patch(circle)

    plt.show()

def tmp():
    """ Temporary function """
    fgpt_ids = [919, 157, 753, 113]
    scenario = "scenario2"
    file_index = 8

    vld_x_r_m = RadioMap()
    vld_x_r_m.load_from_csv('datasets/corrupted/' + scenario + '/ValidationData_' + str(file_index) + '.csv')

    def grade_calculation(centers):
        grade = 0
        for i in range(len(centers)):
            for j in range(i+1, len(centers)):
                grade += np.linalg.norm(centers[i] - centers[j])/len(centers)
        return round(grade, 2)

    for fgpt_id in fgpt_ids:
        fgpt = vld_r_m.get_fingerprint(fgpt_id)
        max_dist, centers = load_ap_max(trning_r_m)
        graph.plot_radio_map(trning_r_m, new_figure=True, alpha=0.3, title="Valid fingerprint : " + str(fgpt_id))
        sorted_centers = []
        for i, rss in enumerate(fgpt.get_rss()):
            if rss != 100:
                if centers[i,0] != 0:
                    plt.scatter(centers[i,0], centers[i,1], marker='x', color='red')
                    sorted_centers.append(centers[i])
        grade = grade_calculation(sorted_centers)
        plt.scatter(fgpt.get_position()[0], fgpt.get_position()[1], marker='o', color='green')
        plt.title("Valid fingerprint : " + str(fgpt_id) + "\nGrade : " + str(grade))


    for fgpt_id in fgpt_ids:
        fgpt = vld_x_r_m.get_fingerprint(fgpt_id)
        max_dist, centers = load_ap_max(trning_r_m)
        graph.plot_radio_map(trning_r_m, new_figure=True, alpha=0.3, title="Corrupted fingerprint : " + str(fgpt_id))
        sorted_centers = []
        for i, rss in enumerate(fgpt.get_rss()):
            if rss != 100:
                if centers[i,0] != 0:
                    plt.scatter(centers[i,0], centers[i,1], marker='x', color='red')
                    sorted_centers.append(centers[i])
        grade = grade_calculation(sorted_centers)
        plt.scatter(fgpt.get_position()[0], fgpt.get_position()[1], marker='o', color='green')
        plt.title("Corrupted fingerprint : " + str(fgpt_id) + "\nGrade : " + str(grade))

    plt.show()
    return



##############################################################################################################

def init():
    """ Initialize the global variables."""
    global trning_r_m, vld_r_m
    print("Loading data...")
    trning_r_m = RadioMap()
    trning_r_m.load_from_csv('datasets/TrainingData.csv')
    vld_r_m = RadioMap()
    vld_r_m.load_from_csv('datasets/ValidationData.csv')
    print("Done !")
    global k_l_values
    k_l_values = {'SC':(8,5), 'UC':(3,15), 'VT':(11,0.64)}


##############################################################################################################

if __name__ == '__main__':
    td = time.time()

    init()

    ##############################

    ## Find the best parameters : SIMU 0X

    # simu01_sc_method()
    # simu01_uc_method()
    # simu01_vt_method()

    # simu02_sc_method()
    # simu02_uc_method()
    # simu02_vt_method()

    # simu02_all()

    ##############################

    ## Test Basic KNN-Algo on Corrupted Data : SIMU 1X

    ## scenario1
    # simu11_scenario1_sc_method()
    # simu11_scenario1_uc_method()
    # simu11_scenario1_vt_method()

    ## scenario2
    # simu12_scenario2_sc_method()
    # simu12_scenario2_uc_method()
    # simu12_scenario2_vt_method()

    ##############################

    ## Test Secure KNN-Algo on Corrupted Data : SIMU 2X

    ## No filter
    # simu20_single_test()

    ## Scenario 1/2

    # Overall filter : SIMU 21
    # simu21_scenario1_uc_method_secu()
    # simu21_scenario2_uc_method_secu()

    # Precise filter : SIMU 22
    # simu22_scenario1_uc_method_secu()
    # simu22_scenario2_uc_method_secu()

    # False Positive Analysis : SIMU 23
    # simu23_overall_filter()
    # simu23_precise_filter()

    # Single tests : SIMU 24
    # simu24_single_test_of()
    # simu23_single_test_pf()

    ##############################

    ## Other simulations

    # display_ap_fingerprints(147)

    # visualize_ap_centers()

    # visualize_ap_centers_vs_fingerprint()

    tmp()

    print("Executed in ", time.time() - td, " seconds")
