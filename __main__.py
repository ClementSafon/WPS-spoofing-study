""" Main file of the project. """
import time
import numpy as np
import graph
import csv
from radio_map import RadioMap
import knn_algorithm as knn
from matplotlib import pyplot as plt
from metadata_gen import load_ap_max

## Simulations

# Find the best parameters : SIMU 01
def simu01_shared_coord_method():
    """ find all the errors for all the K,LIMIT combinations."""
    data = [["LIMIT", "K", "MEAN_ERROR", "STD_ERROR", "FAILRATE", "MAX_ERROR", "MIN_ERROR", "MEDIAN_ERROR", "25th_PERCENTILE", "75th_PERCENTILE", "90th_PERCENTILE", "95th_PERCENTILE", "99th_PERCENTILE", "99.99th_PERCENTILE"]]

    # Custom Input Data
    size_of_the_sample = 100
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
                position_error = knn.find_position_error(k, trning_r_m, vld_r_m.get_fingerprint(fgpt_id), limit)
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
                for k in range(k, k_max):
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
    size_of_the_sample = 100
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
                position_error = knn.find_position_error_UC_method(k, trning_r_m, vld_r_m.get_fingerprint(fgpt_id), limit)
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
                for k in range(k, k_max):
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

    k = 6
    limit = 6

    errors = []
    failed = 0
    for fgpt_id in range(len(vld_r_m)):
        print(round((fgpt_id / len(vld_r_m)) * 100,1), " "*(4-len(str(round((fgpt_id / len(vld_r_m)) * 100,1)))) + "%", end="\r")
        error = knn.find_position_error(k, trning_r_m, vld_r_m.get_fingerprint(fgpt_id), limit)
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
        error = knn.find_position_error_UC_method(k, trning_r_m, vld_r_m.get_fingerprint(fgpt_id), limit)
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

def simu11():
    """ find the mean error for a validation dataset where some rows are alterated."""
    k = 7
    limit = 7

    for file_index in range(1,11):

        vld_X_r_m = RadioMap()
        vld_X_r_m.load('juraj_data/ValidationData_' + str(file_index) + '.csv')


        rows = [i for i in range(len(vld_X_r_m.get_data()))]

        attack_success = 0
        fail_due_to_attack = 0
        fail = 0
        unfail_due_to_attack = 0
        attack_deviations = []
        print('ValidationData_' + str(file_index) + '.csv')
        for row in rows:
            print(str(row) + " / " + str(len(rows)) + " -> " + str(round((row / len(rows)) * 100,2)) + "%", end="\r")
            estimated_position = knn.find_position(k, trning_r_m, vld_X_r_m.fork([row]), limit)
            position = knn.find_position(k, trning_r_m, vld_r_m.fork([row]), limit)
            null_pos = (position == [0,0,0]).all()
            null_est = (estimated_position == [0,0,0]).all()
            if (estimated_position != position).all():
                attack_success += 1
                if not null_est and not null_pos:
                    attack_deviations.append(np.linalg.norm(estimated_position - position))
            if null_est and not null_pos:
                fail_due_to_attack += 1
            if not null_est and null_pos:
                unfail_due_to_attack += 1
            if null_pos:
                fail += 1
            
        print()
        print("Number of attack successful: ", attack_success, "/", len(rows), " -> " + str(round((attack_success / len(rows)) * 100,2))+ "%")
        print("Distance error due to attack -> ", np.mean(attack_deviations))
        print("Number of fail due to attack: ", fail_due_to_attack, "/", len(rows), " -> " + str(round((fail_due_to_attack / len(rows)) * 100,2)) + "%")
        print("unfail due to attack: ", unfail_due_to_attack, "/", len(rows), " -> " + str(round((unfail_due_to_attack / len(rows)) * 100,2)) + "%")
        print("Number of fail: ", fail, "/", len(rows), " -> " + str(round((fail / len(rows)) * 100,2)) + "%")
        print()


def simu2():
    """ Simulate many spoofing scenarios."""
    errors_l = []
    success, fail, total = 0, 0, 0
    log_penality = False
    for i in np.random.randint(0, len(vld_r_m.get_data()), 50):
        for j in np.random.randint(0, len(vld_r_m.get_data()), 100):
            if vld_r_m.get_data()[i]["BUILDINGID"] != vld_r_m.get_data()[j]["BUILDINGID"]:
                print("Spoofing on row ", i, " with row ", j)
                trgt_r_m = vld_r_m.fork([i])
                trgt_r_m.spoof(0, vld_r_m, j)
                pos, dist = knn.run(3, False, False, trning_r_m, trgt_r_m, log_penality)[:2]
                pos_expected = vld_r_m.fork([i]).get_positions()[0]
                pos_fake_APs = vld_r_m.fork([j]).get_positions()[0]
                if (pos[0] != pos_expected).all():
                    dist_to_fake_APs = np.linalg.norm(pos[0] - pos_fake_APs)
                    if dist_to_fake_APs <= confidence_value:
                        success += 1
                    elif dist[0] <= confidence_value:
                        fail += 1
                else:
                    fail += 1
                total += 1
                errors_l.append(dist[0])
    print("Success number: ", success, " / ", total, " (", success/total*100, "%)")
    print("Fail number: ", fail, " / ", total, " (", fail/total*100, "%)")
    print("Partial success number: ", total - success - fail, " / ", total, " (", (
        total - success - fail)/total*100, "%)")
    print("##############-More Info-####################")
    print("Mean error: ", np.mean(errors_l))
    print("Standard deviation: ", np.std(errors_l))
    print("Max error: ", np.max(errors_l))
    print("Min error: ", np.min(errors_l))
    print("Median error: ", np.median(errors_l))
    print("25th percentile: ", np.percentile(errors_l, 25))
    print("75th percentile: ", np.percentile(errors_l, 75))
    print("90th percentile: ", np.percentile(errors_l, 90))
    print("95th percentile: ", np.percentile(errors_l, 95))
    print("99th percentile: ", np.percentile(errors_l, 99))
    print("99.99th percentile: ", np.percentile(errors_l, 99.99))

def simu2_spoofing(spoofed_row, spoofing_row):
    """ Simulate the spoofing."""
    trgt_r_m = vld_r_m.fork([spoofed_row])
    witness_r_m = vld_r_m.fork([spoofed_row])

    log_penality = True

    trgt_r_m.spoof(0, vld_r_m, spoofing_row)
    average_positions, error = knn.run_explicite(
        3, trning_r_m, trgt_r_m, log_penality)[:2]
    witness_average_positions = knn.run_explicite(3, trning_r_m, witness_r_m, log_penality
                            )[0][0]

    title = f"Spoofing with valid fingerprint (on row {spoofed_row})"
    graph.plot_radio_map(trning_r_m, title=title, new_figure=True)
    graph.plot_point(average_positions[0], args='ro', label='Estimated position')
    graph.plot_point(witness_average_positions, args='yo', label='Estimated position without spoofing')
    graph.plot_point(vld_r_m.fork([spoofing_row]).get_positions()[0]
                                , args='ko', label='Position of the point that is used to spoof')
    graph.plot_point(trgt_r_m.get_positions()[0], args='go', label='Real position')
    graph.plot_confidence_circle(average_positions[0], confidence_value)
    graph.show()
    return error[0]

def sim3_randoom_spoofing(spoofed_row, seed):
    """ Simulate the random spoofing."""
    trgt_r_m = vld_r_m.fork([spoofed_row])
    witness_r_m = vld_r_m.fork([spoofed_row])


    trgt_r_m.random_spoof(0, seed)
    average_positions = knn.run(
        3, False, True, trning_r_m, trgt_r_m, log_penality=False)

    title = f"Spoofing with completly random fingerprint (on row {spoofed_row})"
    graph.plot_radio_map(trning_r_m,
                      title=title, new_figure=True)
    graph.plot_point(average_positions[0], args='ro', label='Estimated position')
    graph.plot_point(knn.run(3, False, True, trning_r_m, witness_r_m, log_penality=False
                            )[0][0], args='yo', label='Estimated position without spoofing')
    graph.plot_point(trgt_r_m.get_positions_by_row(0), args='go', label='Real position')
    graph.plot_confidence_circle(average_positions[0], confidence_value)
    graph.show()

def simu4_all_spoofing(spoofed_row, spoofing_row, seed):
    """ Simulate the spoofing."""
    trgt_r_m = vld_r_m.fork([spoofed_row])
    witness_r_m = vld_r_m.fork([spoofing_row])


    trgt_r_m.spoof(spoofed_row, vld_r_m, spoofing_row)
    average_positions = knn.run(
        3, False, True, trning_r_m, trgt_r_m, log_penality=False)[0]

    title = f"Spoofing with valid fingerprint (on row {spoofed_row})"
    graph.plot_radio_map(trning_r_m, title=title, new_figure=True)
    graph.plot_point(average_positions[0], args='ro', label='Estimated position')
    graph.plot_point(knn.run(3, False, True, trning_r_m, witness_r_m, log_penality=False
                            )[0][0], args='yo', label='Estimated position without spoofing')
    graph.plot_point(vld_r_m.fork([spoofing_row]).get_positions_by_row(spoofing_row
                                    ), args='yo', label='Position of the point that is spoofed')
    graph.plot_point(trgt_r_m.get_positions()[0], args='go')
    graph.plot_confidence_circle(average_positions[0], confidence_value)

    trgt_r_m = vld_r_m.fork([spoofed_row])
    witness_r_m = vld_r_m.fork([spoofed_row])


    trgt_r_m.random_spoof(0, seed)
    average_positions = knn.run(
        3, False, True, trning_r_m, trgt_r_m, log_penality=False)

    title = f"Spoofing with completly random fingerprint (on row {spoofed_row})"
    graph.plot_radio_map(trning_r_m,
                      title=title, new_figure=True)
    graph.plot_point(average_positions[0], args='ro', label='Estimated position')
    graph.plot_point(knn.run(3, False, True, trning_r_m, witness_r_m, log_penality=False
                            )[0][0], args='yo', label='Estimated position without spoofing')
    graph.plot_point(trgt_r_m.get_positions_by_row(0), args='go', label='Real position')
    graph.plot_confidence_circle(average_positions[0], confidence_value)
    graph.show()

def simu5():
    """ Simulate attack be re-emitting received beacons """
    """ Simulate many spoofing scenarios."""
    errors_l = []
    success, fail, total = 0, 0, 0
    log_penality = False
    indexes = np.random.randint(0, len(vld_r_m.get_data()), 250)
    trgt_r_m = vld_r_m.fork(indexes)
    for i in range(len(indexes)):
        trgt_r_m.reemitting_spoof(i)
    pos, dist = knn.run(3, False, False, trning_r_m, trgt_r_m, log_penality)[:2]
    pos_expected = vld_r_m.fork(indexes).get_positions()[0]
    for i in range(len(indexes)):
        if (pos[i] != pos_expected).all():
            if dist[i] > confidence_value:
                success += 1
            elif dist[i] <= confidence_value:
                fail += 1
        else:
            fail += 1
        total += 1
        errors_l.append(dist[i])
    print("Success number: ", success, " / ", total, " (", success/total*100, "%)")
    print("Fail number: ", fail, " / ", total, " (", fail/total*100, "%)")
    print("##############-More Info-####################")
    print("Mean error: ", np.mean(errors_l))
    print("Standard deviation: ", np.std(errors_l))
    print("Max error: ", np.max(errors_l))
    print("Min error: ", np.min(errors_l))
    print("Median error: ", np.median(errors_l))
    print("25th percentile: ", np.percentile(errors_l, 25))
    print("75th percentile: ", np.percentile(errors_l, 75))
    print("90th percentile: ", np.percentile(errors_l, 90))
    print("95th percentile: ", np.percentile(errors_l, 95))
    print("99th percentile: ", np.percentile(errors_l, 99))
    print("99.99th percentile: ", np.percentile(errors_l, 99.99))

def simu6_single(spoofed_row=0, spoofing_row=123):
    """ simu6_single one calculation, run_secure knn algo to detect the attack  """
    trgt_r_m = vld_r_m.fork([spoofed_row, spoofed_row])

    trgt_r_m.spoof(0, vld_r_m, spoofing_row)

    knn.run_secure(3, True, trning_r_m, trgt_r_m, method="distance")[:2]

def simu6(method):
    """ same as simu 2 but with knn-secure algo"""
    """ Simulate many spoofing scenarios."""
    errors_l = []
    success, fail, total, detected, partial_success = 0, 0, 0, 0, 0
    for i in np.random.randint(0, len(vld_r_m.get_data()), 50):
        for j in np.random.randint(0, len(vld_r_m.get_data()), 1):
            if vld_r_m.get_data()[i]['BUILDINGID'] != vld_r_m.get_data()[j]['BUILDINGID']:
                print("Spoofing on row ", i, " with row ", j)
                trgt_r_m = vld_r_m.fork([i])
                trgt_r_m.spoof(0, vld_r_m, j)
                pos, dist = knn.run_secure(3, False, trning_r_m, trgt_r_m, method)[:2]
                pos_expected = vld_r_m.fork([i]).get_positions()[0]
                pos_fake_APs = vld_r_m.fork([j]).get_positions()[0]
                if (pos[0] != pos_expected).all():
                    if pos[0][0] == -1:
                        detected += 1
                    else:
                        dist_to_fake_APs = np.linalg.norm(pos[0] - pos_fake_APs)
                        if dist[0] <= confidence_value:
                            fail += 1
                        elif dist_to_fake_APs <= confidence_value:
                            success += 1
                        else:
                            partial_success += 1
                else:
                    fail += 1
                total += 1
                if dist[0] != -1:
                    errors_l.append(dist[0])
    print("Success number: ", success, " / ", total, " (", success/total*100, "%)")
    print("Fail number: ", fail, " / ", total, " (", fail/total*100, "%)")
    print("Partial success number: ", partial_success, " / ", total, " (", partial_success/total*100, "%)")
    print("Detected number: ", detected, " / ", total, " (", detected/total*100, "%)")
    if len(errors_l) > 0:
        print("##############-More Info-####################")
        print("Mean error: ", np.mean(errors_l))
        print("Standard deviation: ", np.std(errors_l))
        print("Max error: ", np.max(errors_l))
        print("Min error: ", np.min(errors_l))
        print("Median error: ", np.median(errors_l))
        print("25th percentile: ", np.percentile(errors_l, 25))
        print("75th percentile: ", np.percentile(errors_l, 75))
        print("90th percentile: ", np.percentile(errors_l, 90))
        print("95th percentile: ", np.percentile(errors_l, 95))
        print("99th percentile: ", np.percentile(errors_l, 99))
        print("99.99th percentile: ", np.percentile(errors_l, 99.99))
    else:
        print("No error")

def simu6_bis(method):
    """ same as simu 2 but with knn-secure algo"""
    """ Simulate many spoofing scenarios."""
    errors_l = []
    total, detected = 0, 0
    for i in np.random.randint(0, len(vld_r_m.get_data()), 50):
        print("Estimating position of row ", i)
        trgt_r_m = vld_r_m.fork([i])
        pos, dist = knn.run_secure(3, False, trning_r_m, trgt_r_m, method)[:2]
        if dist[0] == -1:
            detected += 1
        else:
            errors_l.append(dist[0])
        total += 1
    print("Detected number: ", detected, " / ", total, " (", detected/total*100, "%)")
    if len(errors_l) > 0:
        print("##############-More Info-####################")
        print("Mean error: ", np.mean(errors_l))
        print("Standard deviation: ", np.std(errors_l))
        print("Max error: ", np.max(errors_l))
        print("Min error: ", np.min(errors_l))
        print("Median error: ", np.median(errors_l))
        print("25th percentile: ", np.percentile(errors_l, 25))
        print("75th percentile: ", np.percentile(errors_l, 75))
        print("90th percentile: ", np.percentile(errors_l, 90))
        print("95th percentile: ", np.percentile(errors_l, 95))
        print("99th percentile: ", np.percentile(errors_l, 99))
        print("99.99th percentile: ", np.percentile(errors_l, 99.99))
    else:
        print("No error")

def display_rssi_timestamp():
    """ temporary function to test stuff. """
    # with open('data/ap_max_dist.txt', 'r') as f:
    #     ap_max_dist = np.array([float(dist) for dist in f.readline().strip().split(",")])
    # sorted_indexes = np.argsort(ap_max_dist)
    # print(sorted_indexes[-10:])
    # id_AP = sorted_indexes[-10] + 1
    id_AP = 54
    x_coords, y_coords = [], []
    rssi = []
    timestamp = []
    r_m_rows = []
    for row, fingerprint in enumerate(clean_trning_r_m.get_data()):
        if fingerprint['rss'][id_AP - 1] != 100:
            if fingerprint['LONGITUDE'] in x_coords and fingerprint['LATITUDE'] == y_coords[x_coords.index(fingerprint['LONGITUDE'])]:
                if fingerprint["TIMESTAMP"] > timestamp[x_coords.index(fingerprint['LONGITUDE'])]:
                    rssi[x_coords.index(fingerprint['LONGITUDE'])] = fingerprint['rss'][id_AP - 1]
                    timestamp[x_coords.index(fingerprint['LONGITUDE'])] = fingerprint['TIMESTAMP']
            else:
                x_coords.append(fingerprint['LONGITUDE'])
                y_coords.append(fingerprint['LATITUDE'])
                rssi.append(fingerprint['rss'][id_AP + 1])
                timestamp.append(float(fingerprint['TIMESTAMP']))
        else:
            r_m_rows.append(row)
    r_m = clean_trning_r_m.fork(r_m_rows)

    if len(rssi) == 0:
        print("No data for this AP")
        return

    # Normalize values to a range between 0 and 1
    normalized_rssi = (np.array(rssi) + 110) / 110
    normalized_timestamp = (np.array(timestamp) - np.min(timestamp)) / (np.max(timestamp) - np.min(timestamp))

    # graph.plot_radio_map(r_m)

    # Create a colormap
    colormap = plt.colormaps.get_cmap('plasma')

    # Create a scatter plot
    scatter = plt.scatter(x_coords, y_coords, c=normalized_rssi, cmap=colormap, marker='o', s=60)

    # Add colorbar
    cbar = plt.colorbar(scatter)
    cbar.set_label('RSSI')

    # Set plot labels and title
    plt.xlabel('X Coordinate')
    plt.ylabel('Y Coordinate')
    plt.title('Scatter Plot with Color Gradient')

    plt.figure()
    # graph.plot_radio_map(r_m)
    scatter = plt.scatter(x_coords, y_coords, c=normalized_timestamp, cmap=colormap, marker='o', s=60)

    # Add colorbar
    cbar = plt.colorbar(scatter)
    cbar.set_label('Timestamp')

    # Set plot labels and title
    plt.xlabel('X Coordinate')
    plt.ylabel('Y Coordinate')
    plt.title('Scatter Plot with Color Gradient')

    # Show the plot
    plt.show()

def tmp():
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
    trning_r_m.load_from_csv('data/TrainingData.csv')
    vld_r_m = RadioMap()
    vld_r_m.load_from_csv('data/ValidationData.csv')
    clean_trning_r_m = RadioMap()
    clean_trning_r_m.load_from_csv('data/other/clean_TrainingData.csv')
    print("Done !")

    # simu01_shared_coord_method()
    # simu01_unshared_coord_method()

    simu02_shared_coord_method()
    simu02_unshared_coord_method()

    # Simulation 10
    # simu10()
    # simu10_bis()

    # Simulation 11
    # simu11()

    # display_rssi_timestamp()

    # Simulation 2
    # simu2()
    # simu2_spoofing(290, 654)
    # simu2_spoofing(1100, 650)

    # Simulation 3
    # sim3_randoom_spoofing(0, 123)

    # Simulation 4
    # simu4_all_spoofing(0, 50, 123)

    # Simulation 5
    # simu5()

    # Simulation 6
    # simu6("distance_FA")
    # simu6_bis("distance_FA")
    # simu6_single(0, 123)

    # tmp()

    print("Executed in ", time.time() - td, " seconds")
