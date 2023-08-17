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
    size_of_the_sample = 1000
    k_min = 1
    k_max = 15
    limit_max = 15
    limit_min = 1
    tolerance_fail = 0.1

    fgpt_ids = np.random.randint(0, len(vld_r_m), size_of_the_sample)

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
                print("K=", k, " LIMIT=", limit, " -> ", round(np.median(position_errors),2))
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
    size_of_the_sample = 1000
    k_min = 1
    k_max = 15
    limit_max = 25
    limit_min = 10
    tolerance_fail = 0.1

    fgpt_ids = np.random.randint(0, len(vld_r_m), size_of_the_sample)

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
                print("K=", k, " LIMIT=", limit, " -> ", round(np.median(position_errors),2))
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

# Test the best K and LIMIT
def simu02_shared_coord_method():
    """ find the error for a k, and limit combination."""    

    k = 11
    limit = 7

    errors = []
    failed = 0
    for fgpt_id in range(len(vld_r_m)):
        print(round((fgpt_id / len(vld_r_m)) * 100,1), " "*(4-len(str(round((fgpt_id / len(vld_r_m)) * 100,1)))) + "%", end="\r")
        error = knn.find_position_error(k, trning_r_m, vld_r_m.get_fingerprint(fgpt_id), limit, method="SC")
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

def simu02_unshared_coord_method():
    """ find the error for a k, and limit combination."""    

    k = 7
    limit = 16

    errors = []
    failed = 0
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

def simu02_other_method():
    """ find the error for a k, and limit combination."""    

    k = 7
    limit = 16

    errors = []
    failed = 0
    for fgpt_id in range(len(vld_r_m)):
        print(round((fgpt_id / len(vld_r_m)) * 100,1), " "*(4-len(str(round((fgpt_id / len(vld_r_m)) * 100,1)))) + "%", end="\r")
        error = knn.find_position_error(k, trning_r_m, vld_r_m.get_fingerprint(fgpt_id), limit, method="OT")
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

# Test Basic KNN-Algo on Corrupted Data : SIMU 1X
def simu11_scenario1_SC_method():
    """ find the mean error for a validation dataset where some rows are alterated."""
    k = 11
    limit = 7

    data = [["FILE", "ATTACK_SUCCESSFUL", "ATTACK_FAILED", "POSITIONING_FAILED_DUE_TO_ATTACK", "POSITIONING_NO_FAILED_DUE_TO_ATTACK", "POSITIONING_MEAN_ERROR_DUE_TO_ATTACK"]]

    for file_index in range(1,11):
        vld_X_r_m = RadioMap()
        vld_X_r_m.load_from_csv('datasets/corrupted/scenario1/ValidationData_' + str(file_index) + '.csv')

        n_attack_successfull = 0
        n_attack_failed = 0
        n_positioning_failed_due_to_attack = 0
        n_positioning_no_failed_due_to_attack = 0
        positioning_error_due_to_attack = []
        print('ValidationData_' + str(file_index) + '.csv')
        for fgpt_id in range(len(vld_X_r_m)):
            print(round((fgpt_id / len(vld_r_m)) * 100,1), " "*(4-len(str(round((fgpt_id / len(vld_r_m)) * 100,1)))) + "%", end="\r")
            predicted_position = knn.find_position(k, trning_r_m, vld_X_r_m.get_fingerprint(fgpt_id), limit)
            normal_predicted_position = knn.find_position(k, trning_r_m, vld_r_m.get_fingerprint(fgpt_id), limit)
            null_pred_pos = (predicted_position == [0,0,0]).all()
            null_norm_pos = (normal_predicted_position == [0,0,0]).all()

            if (predicted_position != normal_predicted_position).all():
                n_attack_successfull += 1
                if not null_norm_pos and not null_pred_pos:
                    positioning_error_due_to_attack.append(np.linalg.norm(predicted_position - normal_predicted_position))
            else:
                n_attack_failed += 1
            
            if null_norm_pos and not null_pred_pos:
                n_positioning_no_failed_due_to_attack += 1
            if not null_norm_pos and null_pred_pos:
                n_positioning_failed_due_to_attack += 1
        
        data.append(["ValidationData_" + str(file_index) + '.csv', n_attack_successfull, n_attack_failed, n_positioning_failed_due_to_attack, n_positioning_no_failed_due_to_attack, np.mean(positioning_error_due_to_attack)])
            
    
    csv_file = "results/basic_knn_on_corrupted_dataset_scenario1_using_SC_method_K"+str(k)+"_L"+str(limit)+"_.csv"

    with open(csv_file, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerows(data)

    print(f"CSV file '{csv_file}' created successfully.")

def simu11_scenario1_UC_method():
    """ find the mean error for a validation dataset where some rows are alterated."""
    k = 7
    limit = 16

    data = [["FILE", "ATTACK_SUCCESSFUL", "ATTACK_FAILED", "POSITIONING_FAILED_DUE_TO_ATTACK", "POSITIONING_NO_FAILED_DUE_TO_ATTACK", "POSITIONING_MEAN_ERROR_DUE_TO_ATTACK"]]

    for file_index in range(1,11):
        vld_X_r_m = RadioMap()
        vld_X_r_m.load_from_csv('datasets/corrupted/scenario1/ValidationData_' + str(file_index) + '.csv')

        n_attack_successfull = 0
        n_attack_failed = 0
        n_positioning_failed_due_to_attack = 0
        n_positioning_no_failed_due_to_attack = 0
        positioning_error_due_to_attack = []
        print('ValidationData_' + str(file_index) + '.csv')
        for fgpt_id in range(len(vld_X_r_m)):
            print(round((fgpt_id / len(vld_r_m)) * 100,1), " "*(4-len(str(round((fgpt_id / len(vld_r_m)) * 100,1)))) + "%", end="\r")
            predicted_position = knn.find_position_UC_method(k, trning_r_m, vld_X_r_m.get_fingerprint(fgpt_id), limit)
            normal_predicted_position = knn.find_position_UC_method(k, trning_r_m, vld_r_m.get_fingerprint(fgpt_id), limit)
            null_pred_pos = (predicted_position == [0,0,0]).all()
            null_norm_pos = (normal_predicted_position == [0,0,0]).all()

            if (predicted_position != normal_predicted_position).all():
                n_attack_successfull += 1
                if not null_norm_pos and not null_pred_pos:
                    positioning_error_due_to_attack.append(np.linalg.norm(predicted_position - normal_predicted_position))
            else:
                n_attack_failed += 1
            
            if null_norm_pos and not null_pred_pos:
                n_positioning_no_failed_due_to_attack += 1
            if not null_norm_pos and null_pred_pos:
                n_positioning_failed_due_to_attack += 1
        
        data.append(["ValidationData_" + str(file_index) + '.csv', n_attack_successfull, n_attack_failed, n_positioning_failed_due_to_attack, n_positioning_no_failed_due_to_attack, np.mean(positioning_error_due_to_attack)])
            
    
    csv_file = "results/basic_knn_on_corrupted_dataset_scenario1_using_UC_method_K"+str(k)+"_L"+str(limit)+"_.csv"

    with open(csv_file, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerows(data)

    print(f"CSV file '{csv_file}' created successfully.")

def simu12_scenario2_SC_method():
    """ find the mean error for a validation dataset where some rows are alterated."""
    k = 11
    limit = 7

    data = [["FILE", "ATTACK_SUCCESSFUL", "ATTACK_FAILED", "POSITIONING_FAILED_DUE_TO_ATTACK", "POSITIONING_NO_FAILED_DUE_TO_ATTACK", "POSITIONING_MEAN_ERROR_DUE_TO_ATTACK"]]

    for file_index in range(1,11):
        vld_X_r_m = RadioMap()
        vld_X_r_m.load_from_csv('datasets/corrupted/scenario2/ValidationData_' + str(file_index) + '.csv')

        n_attack_successfull = 0
        n_attack_failed = 0
        n_positioning_failed_due_to_attack = 0
        n_positioning_no_failed_due_to_attack = 0
        positioning_error_due_to_attack = []
        print('ValidationData_' + str(file_index) + '.csv')
        for fgpt_id in range(len(vld_X_r_m)):
            print(round((fgpt_id / len(vld_r_m)) * 100,1), " "*(4-len(str(round((fgpt_id / len(vld_r_m)) * 100,1)))) + "%", end="\r")
            predicted_position = knn.find_position(k, trning_r_m, vld_X_r_m.get_fingerprint(fgpt_id), limit)
            normal_predicted_position = knn.find_position(k, trning_r_m, vld_r_m.get_fingerprint(fgpt_id), limit)
            null_pred_pos = (predicted_position == [0,0,0]).all()
            null_norm_pos = (normal_predicted_position == [0,0,0]).all()

            if (predicted_position != normal_predicted_position).all():
                n_attack_successfull += 1
                if not null_norm_pos and not null_pred_pos:
                    positioning_error_due_to_attack.append(np.linalg.norm(predicted_position - normal_predicted_position))
            else:
                n_attack_failed += 1
            
            if null_norm_pos and not null_pred_pos:
                n_positioning_no_failed_due_to_attack += 1
            if not null_norm_pos and null_pred_pos:
                n_positioning_failed_due_to_attack += 1
        
        data.append(["ValidationData_" + str(file_index) + '.csv', n_attack_successfull, n_attack_failed, n_positioning_failed_due_to_attack, n_positioning_no_failed_due_to_attack, np.mean(positioning_error_due_to_attack)])
            
    
    csv_file = "results/basic_knn_on_corrupted_dataset_scenario2_using_SC_method_K"+str(k)+"_L"+str(limit)+"_.csv"

    with open(csv_file, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerows(data)

    print(f"CSV file '{csv_file}' created successfully.")

def simu12_scenario2_UC_method():
    """ find the mean error for a validation dataset where some rows are alterated."""
    k = 7
    limit = 16

    data = [["FILE", "ATTACK_SUCCESSFUL", "ATTACK_FAILED", "POSITIONING_FAILED_DUE_TO_ATTACK", "POSITIONING_NO_FAILED_DUE_TO_ATTACK", "POSITIONING_MEAN_ERROR_DUE_TO_ATTACK"]]

    for file_index in range(1,11):
        vld_X_r_m = RadioMap()
        vld_X_r_m.load_from_csv('datasets/corrupted/scenario2/ValidationData_' + str(file_index) + '.csv')

        n_attack_successfull = 0
        n_attack_failed = 0
        n_positioning_failed_due_to_attack = 0
        n_positioning_no_failed_due_to_attack = 0
        positioning_error_due_to_attack = []
        print('ValidationData_' + str(file_index) + '.csv')
        for fgpt_id in range(len(vld_X_r_m)):
            print(round((fgpt_id / len(vld_r_m)) * 100,1), " "*(4-len(str(round((fgpt_id / len(vld_r_m)) * 100,1)))) + "%", end="\r")
            predicted_position = knn.find_position_UC_method(k, trning_r_m, vld_X_r_m.get_fingerprint(fgpt_id), limit)
            normal_predicted_position = knn.find_position_UC_method(k, trning_r_m, vld_r_m.get_fingerprint(fgpt_id), limit)
            null_pred_pos = (predicted_position == [0,0,0]).all()
            null_norm_pos = (normal_predicted_position == [0,0,0]).all()

            if (predicted_position != normal_predicted_position).all():
                n_attack_successfull += 1
                if not null_norm_pos and not null_pred_pos:
                    positioning_error_due_to_attack.append(np.linalg.norm(predicted_position - normal_predicted_position))
            else:
                n_attack_failed += 1
            
            if null_norm_pos and not null_pred_pos:
                n_positioning_no_failed_due_to_attack += 1
            if not null_norm_pos and null_pred_pos:
                n_positioning_failed_due_to_attack += 1
        
        data.append(["ValidationData_" + str(file_index) + '.csv', n_attack_successfull, n_attack_failed, n_positioning_failed_due_to_attack, n_positioning_no_failed_due_to_attack, np.mean(positioning_error_due_to_attack)])
            
    
    csv_file = "results/basic_knn_on_corrupted_dataset_scenario2_using_UC_method_K"+str(k)+"_L"+str(limit)+"_.csv"

    with open(csv_file, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerows(data)

    print(f"CSV file '{csv_file}' created successfully.")

def simu13_scenario1_OT_method():
    """ find the mean error for a validation dataset where some rows are alterated."""
    k = 11
    limit = 7

    data = [["FILE", "ATTACK_SUCCESSFUL", "ATTACK_FAILED", "POSITIONING_FAILED_DUE_TO_ATTACK", "POSITIONING_NO_FAILED_DUE_TO_ATTACK", "POSITIONING_MEAN_ERROR_DUE_TO_ATTACK"]]

    for file_index in range(1,11):
        vld_X_r_m = RadioMap()
        vld_X_r_m.load_from_csv('datasets/corrupted/scenario1/ValidationData_' + str(file_index) + '.csv')

        n_attack_successfull = 0
        n_attack_failed = 0
        n_positioning_failed_due_to_attack = 0
        n_positioning_no_failed_due_to_attack = 0
        positioning_error_due_to_attack = []
        print('ValidationData_' + str(file_index) + '.csv')
        for fgpt_id in range(len(vld_X_r_m)):
            print(round((fgpt_id / len(vld_r_m)) * 100,1), " "*(4-len(str(round((fgpt_id / len(vld_r_m)) * 100,1)))) + "%", end="\r")
            predicted_position = knn.find_position_other_method(k, trning_r_m, vld_X_r_m.get_fingerprint(fgpt_id), limit)
            normal_predicted_position = knn.find_position(k, trning_r_m, vld_r_m.get_fingerprint(fgpt_id), limit)
            null_pred_pos = (predicted_position == [0,0,0]).all()
            null_norm_pos = (normal_predicted_position == [0,0,0]).all()

            if (predicted_position != normal_predicted_position).all():
                n_attack_successfull += 1
                if not null_norm_pos and not null_pred_pos:
                    positioning_error_due_to_attack.append(np.linalg.norm(predicted_position - normal_predicted_position))
            else:
                n_attack_failed += 1
            
            if null_norm_pos and not null_pred_pos:
                n_positioning_no_failed_due_to_attack += 1
            if not null_norm_pos and null_pred_pos:
                n_positioning_failed_due_to_attack += 1
        
        data.append(["ValidationData_" + str(file_index) + '.csv', n_attack_successfull, n_attack_failed, n_positioning_failed_due_to_attack, n_positioning_no_failed_due_to_attack, np.mean(positioning_error_due_to_attack)])
            
    
    csv_file = "results/basic_knn_on_corrupted_dataset_scenario1_using_OT_method_K"+str(k)+"_L"+str(limit)+"_.csv"

    with open(csv_file, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerows(data)

    print(f"CSV file '{csv_file}' created successfully.")

def simu13_scenario1_OT_method():
    """ find the mean error for a validation dataset where some rows are alterated."""
    k = 7
    limit = 16

    data = [["FILE", "ATTACK_SUCCESSFUL", "ATTACK_FAILED", "POSITIONING_FAILED_DUE_TO_ATTACK", "POSITIONING_NO_FAILED_DUE_TO_ATTACK", "POSITIONING_MEAN_ERROR_DUE_TO_ATTACK"]]

    for file_index in range(1,11):
        vld_X_r_m = RadioMap()
        vld_X_r_m.load_from_csv('datasets/corrupted/scenario1/ValidationData_' + str(file_index) + '.csv')

        n_attack_successfull = 0
        n_attack_failed = 0
        n_positioning_failed_due_to_attack = 0
        n_positioning_no_failed_due_to_attack = 0
        positioning_error_due_to_attack = []
        print('ValidationData_' + str(file_index) + '.csv')
        for fgpt_id in range(len(vld_X_r_m)):
            print(round((fgpt_id / len(vld_r_m)) * 100,1), " "*(4-len(str(round((fgpt_id / len(vld_r_m)) * 100,1)))) + "%", end="\r")
            predicted_position = knn.find_position_other_method(k, trning_r_m, vld_X_r_m.get_fingerprint(fgpt_id), limit)
            normal_predicted_position = knn.find_position_UC_method(k, trning_r_m, vld_r_m.get_fingerprint(fgpt_id), limit)
            null_pred_pos = (predicted_position == [0,0,0]).all()
            null_norm_pos = (normal_predicted_position == [0,0,0]).all()

            if (predicted_position != normal_predicted_position).all():
                n_attack_successfull += 1
                if not null_norm_pos and not null_pred_pos:
                    positioning_error_due_to_attack.append(np.linalg.norm(predicted_position - normal_predicted_position))
            else:
                n_attack_failed += 1
            
            if null_norm_pos and not null_pred_pos:
                n_positioning_no_failed_due_to_attack += 1
            if not null_norm_pos and null_pred_pos:
                n_positioning_failed_due_to_attack += 1
        
        data.append(["ValidationData_" + str(file_index) + '.csv', n_attack_successfull, n_attack_failed, n_positioning_failed_due_to_attack, n_positioning_no_failed_due_to_attack, np.mean(positioning_error_due_to_attack)])
            
    
    csv_file = "results/basic_knn_on_corrupted_dataset_scenario1_using_OT_method_K"+str(k)+"_L"+str(limit)+"_.csv"

    with open(csv_file, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerows(data)

    print(f"CSV file '{csv_file}' created successfully.")




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



if __name__ == '__main__':
    td = time.time()

    print("Loading data...")
    trning_r_m = RadioMap()
    trning_r_m.load_from_csv('datasets/TrainingData.csv')
    vld_r_m = RadioMap()
    vld_r_m.load_from_csv('datasets/ValidationData.csv')
    print("Done !")

    # simu01_shared_coord_method()
    # simu01_unshared_coord_method()

    # simu02_shared_coord_method()
    # simu02_unshared_coord_method()
    # simu02_other_method()


    simu11_scenario1_SC_method()
    # simu11_scenario1_UC_method()

    # simu12_scenario2_SC_method()
    # simu12_scenario2_UC_method()

    # display_AP_fingerprints(150)

    # tmp()

    print("Executed in ", time.time() - td, " seconds")
