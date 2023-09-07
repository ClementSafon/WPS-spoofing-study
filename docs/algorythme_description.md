# Algorythmes Description

In this part I will discribe the 3 KNN algorythmes that I used in this project.

# KNN - Shared Coordinates

**Input Parameters:**

- `k`: Number of neighbors to consider.
- `trning_r_m` : Training Radio Map (the list of all the training fingerprints).
- `trgt_fgpt` : Target fingerprint (the fingerprint sent by the user).
- `limit` : A threshold to determine the minimum allowable difference in RSS measurements between the target and training fingerprints.

**Data Preparation:**

- `trning_rss_matrix`: The RSS matrix from the training radio map (each line is a training fingerprint RSS vector).
- `trning_pos_matrix` : The position matrix from the training radio map (each line is a training fingerprint position vector in 3D space).
- `trgt_rss` : The RSS vector from the target fingerprint.

**Identify Shared Coordinates:**

- As we know, each fingerprint is composed of a set of RSS measurements. Sometimes some measure can be missing due to the fact that the Access Point is unreachable. In order to compare efffectivly the target fingerprint with the training fingerprints, we need to compare only the training fingerprint that have, in a certain limit, the same RSS measurements as the target fingerprint.

- `shared_coord` : To do so we calculate the **Shared Coordinates** between each the training fingerprint and the target fingerprints. This is done by comparing RSS values to a "no data" value of 100. The goal is to find fingerprints where the difference in RSS measurements is within the specified limit.

- `filtered_trning_rss_matrix` : We filter the training RSS matrix to keep only the fingerprints that have a correcte number of shared coordinates, according to the limit.

**Calculate Squared Differences:**

- `filtered_diff_sq` : We calculate the Squared Differences between the target fingerprint and the filtered training fingerprints.

_**Note:** When a coordinate is missing in one fingerprint compared to the other, we don't take this distance into account (meaning that we add 0 to the sum)._

**Calculate Distances:**

- `distances` : We calculate the euclidian distance between the target fingerprint and the filtered training fingerprints, based on the previous Squared Differences.

**Check Neighbors Count:**

- If the number of distances calculated is less than n_neighbors, it means there are not enough nearby neighbors to estimate the position accurately. In this case, return [0, 0, 0] to indicate that the position estimate is not possible.

**Sort Distances:**

- Sort the distances in ascending order to find the closest neighbors.

**Calculate Average Position:**

- We calculate the average position of the closest neighbors, and return it as the estimated position of the target fingerprint.

## Pseudo Code

```
FUNCTION find_position_SC_method(k, trning_r_m, trgt_fgpt, limit):


    ; Data Preparation

    trning_rss_matrix   ; Extract RSS matrix from the training radio map
    trning_pos_matrix   ; Extract position matrix from the training radio map
    target_rss          ; Extract RSS vector from the target fingerprint


    ; Identify Shared Coordinates AND Calculate Squared Differences

    match_coord_matrix <- SUM OF TRUE WHEN (target_rss != 100) AND (trning_rss_matrix != 100)
    filtered_trning_rss_matrix <- trning_rss_matrix WHERE (match_coord_matrix >= limit)
    filtered_diff_sq <- SQUARE (target_rss - filtered_trning_rss_matrix)

    filtered_trning_pos_matrix <- trning_pos_matrix WHERE (match_coord >= limit)


    ; Adjust Squared Differences

    FOR EACH diff-sq IN filtered_diff_sq :
        IF target_rss == 100 AND trning_rss_matrix[AT diff-sq INDEX] != 100 OR target_rss != 100 AND trning_rss_matrix[AT diff-sq INDEX] == 100 :
            diff <- 0
    END FOR


    ; Calculate Distances

    distances <- SQRT(SUM(filtered_diff_sq))


    ; Check Neighbors Count

    IF LENGHT(distances) < k:
        RETURN (0,0,0)
    END IF


    ; Sort Distances

    sorted_indexes <- SORTED VALUE INDEXES (distances)
    closest_neighbors_indexes <- sorted_indexes[FIRST k VALUES]


    ; Calculate Average Position

    average_position <- MEAN(filtered_trning_pos_matrix[AT closest_neighbors_indexes])


    ; Return Average Position

    RETURN average_position

END FUNCTION
```

# KNN - Unshared Coordinates

This algorythme is the same as the previous one, except that we don't filter the training fingerprints by the number of common coordinates, but by the number of values that are in one fingerprint and not in the other.

The differences are the following:

**Input Parameters:**

- `limit` : A threshold to determine the maximum allowable difference in RSS measurements between the target and training fingerprints.

_Note: The limit is now a maximum instead of a minimum._

**Calculate Squared Differences:**

- `filtered_diff_sq` : We calculate the Squared Differences between the target fingerprint and the filtered training fingerprints.

**Adjust Squared Differences:**

- Now, instead of no taking into account the difference when one fingerprint has a missing value compared to the other, now to improve the precision of the algorythme, we add a penalty. The penalty is proportional to the number of unshared coordinates (RSS values that are in only one fingerprint). We have choosen to add 13 times the number of unshared coordinates.

## Pseudo Code

```
FUNCTION find_position_SC_method(k, trning_r_m, trgt_fgpt, limit):


    ; Data Preparation

    trning_rss_matrix   ; Extract RSS matrix from the training radio map
    trning_pos_matrix   ; Extract position matrix from the training radio map
    target_rss          ; Extract RSS vector from the target fingerprint


    ; Identify Shared Coordinates AND Calculate Squared Differences

    unmatch_coord_matrix = SUM OF TRUE WHEN ((target_rss == 100) AND (trning_rss_matrix != 100)) OR ((target_rss != 100) AND (trning_rss_matrix == 100))

    filtered_trning_rss_matrix <- trning_rss_matrix WHERE (match_coord_matrix < limit)
    filtered_diff_sq <- SQUARE (target_rss - filtered_trning_rss_matrix)

    filtered_trning_pos_matrix <- trning_pos_matrix WHERE (match_coord < limit)


    ; Adjust Squared Differences

    FOR EACH diff-sq IN filtered_diff_sq :
        IF target_rss == 100 AND trning_rss_matrix[AT diff-sq INDEX] != 100 OR target_rss != 100 AND trning_rss_matrix[AT diff-sq INDEX] == 100 :
            diff <- 13*unmatch_coord[AT diff-sq INDEX]
    END FOR


    ; Calculate Distances

    distances <- SQRT(SUM(filtered_diff_sq))


    ; Check Neighbors Count

    IF LENGHT(distances) < k:
        RETURN (0,0,0)
    END IF


    ; Sort Distances

    sorted_indexes <- SORTED VALUE INDEXES (distances)
    closest_neighbors_indexes <- sorted_indexes[FIRST k VALUES]


    ; Calculate Average Position

    average_position <- MEAN(filtered_trning_pos_matrix[AT closest_neighbors_indexes])


    ; Return Average Position

    RETURN average_position

END FUNCTION
```

# KNN - Variable Threshold (VT)

This algorythme is the same as the previous one, except that we don't set a fixed limit to filter the training fingerprints, but we calculate a limit for each fingerprint based on the number of shared coordinates compared to the number of total valid values.

The differences are the following:

**Input Parameters:**

- `limit_rate` : A threshold to determine the minimum rate of shared coordinates between the target and training fingerprints.

_Note: The limit is now a rate instead of a number._

**Calculate the limit:**

- `limit` : We calculate the limit for each fingerprint based on the limit_rate and the number of valid RSS values in the target fingerprint.

_Note: Here, we've took a "valid" RSS value if the value is not 100 and >= -85 dBm._

**Adjust Squared Differences:**

- As the UC method, we add a penalty when needed to improve the accuracy of the algorythme.

## Pseudo Code

```
FUNCTION find_position_SC_method(k, trning_r_m, trgt_fgpt, limit):


    ; Data Preparation

    trning_rss_matrix   ; Extract RSS matrix from the training radio map
    trning_pos_matrix   ; Extract position matrix from the training radio map
    target_rss          ; Extract RSS vector from the target fingerprint


    ; Identify Shared Coordinates AND Calculate Squared Differences

    match_coord_matrix <- SUM OF TRUE WHEN (target_rss != 100) AND (trning_rss_matrix != 100)
    unmatch_coord_matrix = SUM OF TRUE WHEN ((target_rss == 100) AND (trning_rss_matrix != 100)) OR ((target_rss != 100) AND (trning_rss_matrix == 100))

    limit <- limit_rate * SUM OF TRUE WHEN (target_rss != 100) AND (target_rss >= -85)

    filtered_trning_rss_matrix <- trning_rss_matrix WHERE (match_coord_matrix >= limit)
    filtered_diff_sq <- SQUARE (target_rss - filtered_trning_rss_matrix)

    filtered_trning_pos_matrix <- trning_pos_matrix WHERE (match_coord >= limit)


    ; Adjust Squared Differences

    FOR EACH diff-sq IN filtered_diff_sq :
        IF target_rss == 100 AND trning_rss_matrix[AT diff-sq INDEX] != 100 OR target_rss != 100 AND trning_rss_matrix[AT diff-sq INDEX] == 100 :
            diff <- 13*unmatch_coord[AT diff-sq INDEX]
    END FOR


    ; Calculate Distances

    distances <- SQRT(SUM(filtered_diff_sq))


    ; Check Neighbors Count

    IF LENGHT(distances) < k:
        RETURN (0,0,0)
    END IF


    ; Sort Distances

    sorted_indexes <- SORTED VALUE INDEXES (distances)
    closest_neighbors_indexes <- sorted_indexes[FIRST k VALUES]


    ; Calculate Average Position

    average_position <- MEAN(filtered_trning_pos_matrix[AT closest_neighbors_indexes])


    ; Return Average Position

    RETURN average_position

END FUNCTION
```

# KNN - Secure - How to detect an attack

In this part I will explain a way to detect an spoofing attack. The idea of the attack is to send fake beacons as if they were sent by Access Point that are not supposed to be here (where the user is). So the determination of the position could be less precise, or even, could be forced to give to the user an other position wanted by the attacker.

The first way to try to counter thoses attacks is to try to detect them. To do so, we can make a filter for the user fingerprint, in order to exclude the corrupted ones. The idea is to check which AP are detectd on the fingerprint, and try to detect if there are some AP that are not supposed to be here because they supposed to be too far from other AP that are detected.

First we have to generate some metadata from the training dataset. For each AP in the Training Dataset, we will take the average weighted position of the fingerprints, and the max distances between the fingerprints.

Base on those metadata, we can have a map with circles for every AP. Finally, to detect an attack, we can check if every AP detected in the user fingerprint share a common area on the map. If not, we can consider that there is an attack.

To go a little bit futher, we can set threashold to allow some circle to not share the same area. We can do that with two threshold for example, a global one, and a local one. The global one will count the number of AP that are not sharing a common area 2 by 2, and the local one will count the number of AP that are not sharing the same area for each AP. It the same way of counting but not the same way of fitering. In the first case the threshold is global and in the second case we submit the treshold for each AP.
