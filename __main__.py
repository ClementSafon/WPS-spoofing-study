""" Main file of the project. """
import time
import numpy as np
import graph
import csv
from radio_map import RadioMap
import knn_algorithm as knn
from matplotlib import pyplot as plt
from metadata_gen import load_ap_max
import matplotlib as mpl

## Simulations

# Find the best parameters : SIMU 0X
def simu01_shared_coord_method():
    """ find all the errors for all the K,LIMIT combinations."""
    data = [["LIMIT", "K", "MEAN_ERROR", "STD_ERROR", "FAILRATE", "MAX_ERROR", "MIN_ERROR", "MEDIAN_ERROR", "25th_PERCENTILE", "75th_PERCENTILE", "90th_PERCENTILE", "95th_PERCENTILE", "99th_PERCENTILE", "99.99th_PERCENTILE"]]

    # Custom Input Data
    size_of_the_sample = len(vld_r_m)
    k_min = 1
    k_max = 15
    limit_max = 15
    limit_min = 1
    tolerance_fail = 0.1

    fgpt_ids = [i for i in range(0, len(vld_r_m))]

    for limit in range(limit_max, limit_min - 1, -1):
        for k in range(k_min, k_max + 1):
            position_errors = []
            count_fail = 0
            for i, fgpt_id in enumerate(fgpt_ids):
                print(round((i / len(fgpt_ids)) * 100,2), " "*(4-len(str(round((i / len(fgpt_ids)) * 100,0)))) + "%", end="\r")
                position_error = knn.find_position_error(k, trning_r_m, vld_r_m.get_fingerprint(fgpt_id), limit, method="SC")
                if position_error == np.inf:
                    count_fail += 1
                else:
                    position_errors.append(position_error)
                if count_fail / len(fgpt_ids) > tolerance_fail:
                    position_errors = []
                    break
            if len(position_errors) > 0:
                data.append([limit, k, np.mean(position_errors), np.std(position_errors), count_fail/len(fgpt_ids),
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

def simu01_unshared_coord_method():
    """ find all the errors for all the K,LIMIT combinations."""
    data = [["LIMIT", "K", "MEAN_ERROR", "STD_ERROR", "FAILRATE", "MAX_ERROR", "MIN_ERROR", "MEDIAN_ERROR", "25th_PERCENTILE", "75th_PERCENTILE", "90th_PERCENTILE", "95th_PERCENTILE", "99th_PERCENTILE", "99.99th_PERCENTILE"]]

    # Custom Input Data
    size_of_the_sample = len(vld_r_m)
    k_min = 1
    k_max = 15
    limit_max = 25
    limit_min = 10
    tolerance_fail = 0.1

    fgpt_ids = [i for i in range(0, len(vld_r_m))]

    for limit in range(limit_min, limit_max + 1):
        for k in range(k_min, k_max + 1):
            position_errors = []
            count_fail = 0
            for i, fgpt_id in enumerate(fgpt_ids):
                print(round((i / len(fgpt_ids)) * 100,2), " "*(4-len(str(round((i / len(fgpt_ids)) * 100,0)))) + "%", end="\r")
                position_error = knn.find_position_error(k, trning_r_m, vld_r_m.get_fingerprint(fgpt_id), limit, method="UC")
                if position_error == np.inf:
                    count_fail += 1
                else:
                    position_errors.append(position_error)
                if count_fail / len(fgpt_ids) > tolerance_fail:
                    position_errors = []
                    break
            if len(position_errors) > 0:
                data.append([limit, k, np.mean(position_errors), np.std(position_errors), count_fail/len(fgpt_ids),
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

def simu01_variable_threshold_method():
    """ find all the errors for all the K,LIMIT combinations."""
    data = [["LIMIT", "K", "MEAN_ERROR", "STD_ERROR", "FAILRATE", "MAX_ERROR", "MIN_ERROR", "MEDIAN_ERROR", "25th_PERCENTILE", "75th_PERCENTILE", "90th_PERCENTILE", "95th_PERCENTILE", "99th_PERCENTILE", "99.99th_PERCENTILE"]]

    # Custom Input Data
    size_of_the_sample = len(vld_r_m)
    k_min = 1
    k_max = 15
    limit_max = 0.8
    limit_min = 0.5
    limit_pas = 0.01
    tolerance_fail = 0.1

    fgpt_ids = [i for i in range(0, len(vld_r_m))]

    for limit in np.linspace(limit_min, limit_max, int((limit_max - limit_min) / limit_pas) + 1):
        for k in range(k_min, k_max + 1):
            position_errors = []
            count_fail = 0
            for i, fgpt_id in enumerate(fgpt_ids):
                print(round((i / len(fgpt_ids)) * 100,2), " "*(4-len(str(round((i / len(fgpt_ids)) * 100,0)))) + "%", end="\r")
                position_error = knn.find_position_error(k, trning_r_m, vld_r_m.get_fingerprint(fgpt_id), limit, method="VT")
                if position_error == np.inf:
                    count_fail += 1
                else:
                    position_errors.append(position_error)
                if count_fail / len(fgpt_ids) > tolerance_fail:
                    position_errors = []
                    break
            if len(position_errors) > 0:
                data.append([limit, k, np.mean(position_errors), np.std(position_errors), count_fail/len(fgpt_ids),
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


############################################################################################################

# Test the best K and LIMIT
def simu02_shared_coord_method():
    """ find the error for a k, and limit combination."""    

    k = 11
    limit = 7

    errors = []
    failed = 0
    sum_durations = 0
    for fgpt_id in range(len(vld_r_m)):
        print(round((fgpt_id / len(vld_r_m)) * 100,1), " "*(4-len(str(round((fgpt_id / len(vld_r_m)) * 100,1)))) + "%", end="\r")
        error = knn.find_position_error(k, trning_r_m, vld_r_m.get_fingerprint(fgpt_id), limit, method="SC")
        sum_durations += knn.duration
        if error != np.inf:
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

def simu02_unshared_coord_method():
    """ find the error for a k, and limit combination."""    

    k = 7
    limit = 16

    errors = []
    failed = 0
    sum_durations = 0
    for fgpt_id in range(len(vld_r_m)):
        print(round((fgpt_id / len(vld_r_m)) * 100,1), " "*(4-len(str(round((fgpt_id / len(vld_r_m)) * 100,1)))) + "%", end="\r")
        error = knn.find_position_error(k, trning_r_m, vld_r_m.get_fingerprint(fgpt_id), limit, method="UC")
        if error != np.inf:
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

def simu02_variable_threshold_method():
    """ find the error for a k, and limit combination."""    

    k = 11
    limit = 0.64

    errors = []
    failed = 0
    sum_durations = 0
    for fgpt_id in range(len(vld_r_m)):
        print(round((fgpt_id / len(vld_r_m)) * 100,1), " "*(4-len(str(round((fgpt_id / len(vld_r_m)) * 100,1)))) + "%", end="\r")
        error = knn.find_position_error(k, trning_r_m, vld_r_m.get_fingerprint(fgpt_id), limit, method="VT")
        if error != np.inf:
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

def simu02_overall():
    """ find the performance for each methods. """

    methods = ["SC", "UC", "VT", "SECU"]
    k_l_values = [(11,7), (7,16), (11,0.64), (7,16)]

    data = [["method", "(k,l)", "mean_error", "std_error", "max_error", "min_error", "median_error", "failed", "time_to_compute_one_position"]]

    for method, k_l_value in zip(methods, k_l_values):
        errors = []
        failed = 0
        sum_durations = 0
        print("Method: ", method)
        for fgpt_id in range(len(vld_r_m)):
            print(round((fgpt_id / len(vld_r_m)) * 100,1), " "*(4-len(str(round((fgpt_id / len(vld_r_m)) * 100,1)))) + "%", end="\r")
            error = knn.find_position_error(k_l_value[0], trning_r_m, vld_r_m.get_fingerprint(fgpt_id), k_l_value[1], method=method)
            sum_durations += knn.duration
            if error != np.inf:
                errors.append(error)
            else:
                failed += 1
        
        data.append([method, k_l_value, np.mean(errors), np.std(errors), np.max(errors), np.min(errors), np.median(errors), round(failed*100/len(vld_r_m),2), round((sum_durations/len(vld_r_m))*1000,2)])

    csv_file = "results/K_L_overall_performance_evaluations.csv"

    with open(csv_file, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerows(data)
    
    print(f"CSV file '{csv_file}' created successfully.")

############################################################################################################

# Test Basic KNN-Algo on Corrupted Data : SIMU 1X
def simu1X_scenarioX_XX_method(k: int, limit: float, scenario: str, method: str, find: callable):
    data = [["FILE", "ATTACK_SUCCESSFUL", "ATTACK_FAILED", "POSITIONING_FAIL", "NORMAL_POSITIONING_FAIL", "MEAN_ERROR_NORMAL_RSS", "MEAN_ERROR_ACTUAL_POSITION", "TOTAL_OF_ATTACK"]]

    for file_index in range(1,11):
        vld_X_r_m = RadioMap()
        vld_X_r_m.load_from_csv('datasets/corrupted/' + scenario + '/ValidationData_' + str(file_index) + '.csv')

        n_attack_successfull = 0
        n_attack_failed = 0
        positioning_failed = 0
        normal_positioning_failed = 0
        distance_error_normal_rss = []
        distance_error_actual_position = []
        total_of_attack = len(vld_X_r_m)
        print('ValidationData_' + str(file_index) + '.csv')
        for fgpt_id in range(len(vld_X_r_m)):
            print(round((fgpt_id / len(vld_r_m)) * 100,1), " "*(4-len(str(round((fgpt_id / len(vld_r_m)) * 100,1)))) + "%", end="\r")
            predicted_position = find(k, trning_r_m, vld_X_r_m.get_fingerprint(fgpt_id), limit)
            normal_predicted_position = find(k, trning_r_m, vld_r_m.get_fingerprint(fgpt_id), limit)
            actual_position = vld_X_r_m.get_position(fgpt_id)
            null_pred_pos = (predicted_position == [0,0,0]).all()
            null_norm_pos = (normal_predicted_position == [0,0,0]).all()

            if null_norm_pos:
                normal_positioning_failed += 1
            if null_pred_pos:
                positioning_failed += 1
                n_attack_failed += 1
            else:
                if (predicted_position != normal_predicted_position).all():
                    n_attack_successfull += 1
                    if not null_norm_pos and not null_pred_pos:
                        distance_error_normal_rss.append(np.linalg.norm(predicted_position - normal_predicted_position))
                        distance_error_actual_position.append(np.linalg.norm(predicted_position - vld_X_r_m.get_position(fgpt_id)))
                else:
                    n_attack_failed += 1
                    
        data.append(["ValidationData_" + str(file_index) + '.csv', n_attack_successfull, n_attack_failed, positioning_failed, normal_positioning_failed, np.mean(distance_error_normal_rss), np.mean(distance_error_actual_position), total_of_attack])
            
    csv_file = "results/basic_knn_on_corrupted_dataset_" + scenario + "_using_" + method + "_method_K"+str(k)+"_L"+str(limit)+"_.csv"

    with open(csv_file, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerows(data)
    print(f"CSV file '{csv_file}' created successfully.")

# Scenario 1
def simu11_scenario1_SC_method():
    """ find the mean error for a validation dataset where some rows are alterated."""
    k = 11
    limit = 7
    scenario = "scenario1"
    method = "SC"
    find = knn.find_position_SC_method

    simu1X_scenarioX_XX_method(k, limit, scenario, method, find)

def simu11_scenario1_UC_method():
    """ find the mean error for a validation dataset where some rows are alterated."""
    k = 7
    limit = 16
    scenario = "scenario1"
    method = "UC"
    find = knn.find_position_UC_method

    simu1X_scenarioX_XX_method(k, limit, scenario, method, find)

def simu11_scenario1_VT_method():
    """ find the mean error for a validation dataset where some rows are alterated."""
    k = 4
    limit = 0.6
    scenario = "scenario1"
    method = "VT"
    find = knn.find_position_VT_method

    simu1X_scenarioX_XX_method(k, limit, scenario, method, find)
    
# Scenario 2
def simu12_scenario2_SC_method():
    """ find the mean error for a validation dataset where some rows are alterated."""
    k = 11
    limit = 7
    scenario = "scenario2"
    method = "SC"
    find = knn.find_position_SC_method

    simu1X_scenarioX_XX_method(k, limit, scenario, method, find)

def simu12_scenario2_UC_method():
    """ find the mean error for a validation dataset where some rows are alterated."""
    k = 7
    limit = 16
    scenario = "scenario2"
    method = "UC"
    find = knn.find_position_UC_method

    simu1X_scenarioX_XX_method(k, limit, scenario, method, find)

def simu12_scenario2_VT_method():
    """ find the mean error for a validation dataset where some rows are alterated."""
    k = 4
    limit = 0.6
    scenario = "scenario2"
    method = "VT"
    find = knn.find_position_VT_method

    simu1X_scenarioX_XX_method(k, limit, scenario, method, find)


#############################################################################################################

# Security tests
def simu2X_scenarioX_XX_method(k: int, limit: float, scenario: str, method: str, find: callable):
    data = [["FILE", "ATTACK_SUCCESSFUL", "ATTACK_FAILED", "POSITIONING_FAIL", "NORMAL_POSITIONING_FAIL", "MEAN_ERROR_NORMAL_RSS", "MEAN_ERROR_ACTUAL_POSITION", "TOTAL_OF_ATTACK"]]

    for file_index in range(1,11):
        vld_X_r_m = RadioMap()
        vld_X_r_m.load_from_csv('datasets/corrupted/' + scenario + '/ValidationData_' + str(file_index) + '.csv')

        n_attack_successfull = 0
        n_attack_failed = 0
        positioning_failed = 0
        normal_positioning_failed = 0
        distance_error_normal_rss = []
        distance_error_actual_position = []
        total_of_attack = 0
        print('ValidationData_' + str(file_index) + '.csv')
        for fgpt_id in range(len(vld_X_r_m)):
            print(round((fgpt_id / len(vld_r_m)) * 100,1), " "*(4-len(str(round((fgpt_id / len(vld_r_m)) * 100,1)))) + "%", end="\r")
            predicted_position = find(k, trning_r_m, vld_X_r_m.get_fingerprint(fgpt_id), limit)
            normal_predicted_position = find(k, trning_r_m, vld_r_m.get_fingerprint(fgpt_id), limit)
            actual_position = vld_X_r_m.get_position(fgpt_id)
            null_pred_pos = (predicted_position == [0,0,0]).all()
            null_norm_pos = (normal_predicted_position == [0,0,0]).all()

            
            if null_norm_pos:
                normal_positioning_failed += 1
            if null_pred_pos:
                positioning_failed += 1
                n_attack_failed += 1
            else:
                if (predicted_position != normal_predicted_position).all():
                    n_attack_successfull += 1
                    if not null_norm_pos and not null_pred_pos:
                        distance_error_normal_rss.append(np.linalg.norm(predicted_position - normal_predicted_position))
                        distance_error_actual_position.append(np.linalg.norm(predicted_position - vld_X_r_m.get_position(fgpt_id)))
                else:
                    n_attack_failed += 1
                    

            total_of_attack = len(vld_X_r_m)
                    
        data.append(["ValidationData_" + str(file_index) + '.csv', n_attack_successfull, n_attack_failed, positioning_failed, normal_positioning_failed, np.mean(distance_error_normal_rss), np.mean(distance_error_actual_position), total_of_attack])
            
    csv_file = "results/secure_knn_on_corrupted_dataset_" + scenario + "_using_" + method + "_method_K"+str(k)+"_L"+str(limit)+"_.csv"

    with open(csv_file, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerows(data)
    print(f"CSV file '{csv_file}' created successfully.")

def simu21_scenario1_UC_method_secu():
    """ find the mean error for a validation dataset where some rows are alterated."""
    k = 7
    limit = 16
    scenario = "scenario1"
    method = "UC"
    find = knn.find_position_secure

def simu21_scenario2_UC_method_secu():
    """ find the mean error for a validation dataset where some rows are alterated."""
    k = 7
    limit = 16
    scenario = "scenario2"
    method = "UC"
    find = knn.find_position_secure

    simu2X_scenarioX_XX_method(k, limit, scenario, method, find)

def simu22_UC_method_secu():
    """ find the error for a k, and limit combination."""    

    k = 7
    limit = 16

    errors = []
    failed = 0
    for fgpt_id in range(len(vld_r_m)):
        print(round((fgpt_id / len(vld_r_m)) * 100,1), " "*(4-len(str(round((fgpt_id / len(vld_r_m)) * 100,1)))) + "%", end="\r")
        error = knn.find_position_error(k, trning_r_m, vld_r_m.get_fingerprint(fgpt_id), limit, method="SECU")
        if error != np.inf:
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

def tmp():
    """ Temporary function. """
    removed_fingerprints = []
    for tol_i, tol in enumerate(np.linspace(0, 1, 10)):
        # show the progress
        print(tol*100, "%", end="\r")
        removed_fingerprints.append([])
        for ap in range(len(trning_r_m.get_data()[0]['rss'])):
            timestamp = np.zeros(len(trning_r_m.get_data()))
            for row, fingerprint in enumerate(trning_r_m.get_data()):
                if fingerprint['rss'][ap] != 100:
                    timestamp[row] = fingerprint['TIMESTAMP']
            remove_indexes = np.where(timestamp == 0)[0]
            timestamp = np.delete(timestamp, remove_indexes)
            r_i = np.argsort(timestamp)[int(len(timestamp) * tol):]
            for i in r_i:
                if i not in removed_fingerprints[tol_i]:
                    removed_fingerprints[tol_i].append(i)
        removed_fingerprints[tol_i] = len(removed_fingerprints[tol_i]) / len(trning_r_m.get_data())*100
    plt.plot(np.linspace(0, 1, 10), np.array(removed_fingerprints))
    plt.xlabel("Tolerance")
    plt.ylabel("Percentage of removed fingerprints")
    plt.title("Percentage of removed fingerprints as a function of the tolerance")
    plt.show()

def tmp2():
    """ find the minimal error we can have with K=11"""    

    k = 11
    



##############################################################################################################

if __name__ == '__main__':
    td = time.time()

    print("Loading data...")
    trning_r_m = RadioMap()
    trning_r_m.load_from_csv('datasets/TrainingData.csv')
    vld_r_m = RadioMap()
    vld_r_m.load_from_csv('datasets/ValidationData.csv')
    print("Done !")

    simu01_shared_coord_method()
    simu01_unshared_coord_method()
    simu01_variable_threshold_method()

    ##############################

    # simu02_shared_coord_method()
    # simu02_unshared_coord_method()
    # simu02_variable_threshold_method()
    # simu02_overall()

    ##############################

    #scenario1
    # simu11_scenario1_SC_method()
    # simu11_scenario1_UC_method()
    # simu11_scenario1_VT_method()

    #scenario2
    # simu12_scenario2_SC_method()
    # simu12_scenario2_UC_method()
    # simu12_scenario2_VT_method()

    ##############################
    
    # Security tests
    # simu21_scenario1_UC_method_secu()
    # simu21_scenario2_UC_method_secu()

    # simu22_UC_method_secu()

    # display_AP_fingerprints(243)

    # tmp()
    # tmp2()

    print("Executed in ", time.time() - td, " seconds")
