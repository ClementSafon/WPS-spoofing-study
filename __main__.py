""" Main file of the project. """
import time
import numpy as np
import graph
import csv
from radio_map import RadioMap
import knn_algorithm as knn
from matplotlib import pyplot as plt

def simu1_display(error_positions, error_floors):
    """ Display the results of the simulation 1."""
    print(" -> location:")
    print("# Mean error: ", np.mean(error_positions))
    print("# Standard deviation: ", np.std(error_positions))
    print("# Max error: ", np.max(error_positions))
    print("# Min error: ", np.min(error_positions))
    print("# Median error: ", np.median(error_positions))
    print("# 25th percentile: ", np.percentile(error_positions, 25))
    print("# 75th percentile: ", np.percentile(error_positions, 75))
    print("# 90th percentile: ", np.percentile(error_positions, 90))
    print("# 95th percentile: ", np.percentile(error_positions, 95))
    print("# 99th percentile: ", np.percentile(error_positions, 99))
    print("# 99.99th percentile: ", np.percentile(error_positions, 99.99))
    print(" -> Floor:")
    print("# Mean error: ", np.mean(error_floors))
    print("# Standard deviation: ", np.std(error_floors))
    print("# Max error: ", np.max(error_floors))
    print("# Min error: ", np.min(error_floors))
    print("# Median error: ", np.median(error_floors))

def simu10_find_error(n_neighbors, limit, row):
    """ Find the mean error for a given number of neighbors and limit on the position of a row."""
    trgt_r_m = vld_r_m.fork([row])
    predicted_position = knn.find_position(n_neighbors, trning_r_m, trgt_r_m, limit)
    # predicted_position = knn.find_position_UC_method(n_neighbors, trning_r_m, trgt_r_m, limit)
    if (predicted_position == [0,0,0]).all():
        return np.inf
    actual_2d_position = trgt_r_m.get_positions()[0]
    actual_floor = trgt_r_m.get_floors()[0]
    floor_height = 3.0
    actual_position = np.array([actual_2d_position[0], actual_2d_position[1], actual_floor * floor_height])
    predicted_position = np.array([predicted_position[0], predicted_position[1], predicted_position[2] * floor_height])
    error_position = np.linalg.norm(
        predicted_position - actual_position)
    return error_position

def simu10():
    """ find all the errors for all the K,LIMIT combinations."""
    data = [["LIMIT", "K", "MEAN_ERROR", "STD_ERROR", "MAX_ERROR", "MIN_ERROR", "MEDIAN_ERROR", "25th_PERCENTILE", "75th_PERCENTILE", "90th_PERCENTILE", "95th_PERCENTILE", "99th_PERCENTILE", "99.99th_PERCENTILE"]]

    rows = np.random.randint(0, len(vld_r_m.get_data()), 75)
    k_max = 30
    k_min = 1
    limit_max = 20
    limit_min = 1

    tolerance_fail = 0.1

    for limit in range(limit_max, limit_min - 1, -1):
        for k in range(k_min, k_max + 1):
            errors = []
            fail = 0
            for i, row in enumerate(rows):
                print(round((i / len(rows)) * 100,2), "%         ", end="\r")
                error = simu10_find_error(k, limit, row)
                if error != np.inf:
                    errors.append(error)
                else:
                    fail += 1
                if fail / len(rows) > tolerance_fail:
                    errors = []
                    break
            if len(errors) > 0:
                data.append([limit, k, np.mean(errors), np.std(errors), np.max(errors), np.min(errors), np.median(errors), np.percentile(errors, 25), np.percentile(errors, 75), np.percentile(errors, 90), np.percentile(errors, 95), np.percentile(errors, 99), np.percentile(errors, 99.99)])
                print("K=", k, " LIMIT=", limit, " -> ", round(np.mean(errors),2))
            else:
                print("K=", k, " LIMIT=", limit, " -> ", "x                                  ")
                print("...")
                for k in range(k, k_max):
                    data.append([limit, k, "x", "x", "x", "x", "x", "x", "x", "x", "x", "x", "x"])  
                break

    csv_file = "output.csv"

    with open(csv_file, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerows(data)

    print(f"CSV file '{csv_file}' created successfully.")

def simu10_bis():
    """ find the coverage for a k, and limit combination."""
    k = 7
    limit = 7

    rows = [i for i in range(len(vld_r_m.get_data()))]
    errors = []
    failed = 0
    print('Total: ', len(rows))
    for row in rows:
        print(str(row) + " / " + str(len(rows)) + " -> " + str(round((row / len(rows)) * 100,2)) + "%", end="\r")
        error = simu10_find_error(k, limit, row)
        if error != np.inf:
            errors.append(error)
        else:
            failed += 1
    
    print("K=", k, " LIMIT=", limit, " -> (mean error) ", np.mean(errors), " (std error) ", np.std(errors), " (max error) ", np.max(errors), " (min error) ", np.min(errors), " (median error) ", np.median(errors))
    print("Failed: ", failed, "/", len(rows), " -> ", round(failed*100 / len(rows),2))


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

def tmp():
    test_r_m = RadioMap()
    test_r_m.load('data/ValidationData.csv')
    n = []
    for rss in test_r_m.get_rss():
        n.append(np.sum(rss != 100))
    print(np.mean(n))

if __name__ == '__main__':
    td = time.time()

    confidence_value = 24.42282

    trning_r_m = RadioMap()
    trning_r_m.load('data/TrainingData.csv')
    vld_r_m = RadioMap()
    vld_r_m.load('data/ValidationData.csv')

    # Simulation 10
    # simu10()
    simu10_bis()

    # Simulation 11
    # simu11()

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
