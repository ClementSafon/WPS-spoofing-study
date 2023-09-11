# WPS - Study on WiFi based Spoofing attacks

## Description

This repository contains a KNN positioning algorithm based on WiFi fingerprints, used to simulate localization requests and spoofing attacks.

In terms of positioning services, GPS can sometimes lack precision, making WiFi-based positioning services common in indoor localization systems. This approach relies on the Received Signal Strength (RSS) of beacons emitted by WiFi access points.

However, beacons are not secure and can be spoofed by an attacker. An attacker can send fake beacons, making them appear as if they were from Access Points (APs) that are not actually present. This could result in the user being localized incorrectly.

In this repository, we aim to implement various KNN algorithms and evaluate their performance in terms of accuracy and robustness against spoofing attacks.

## The KNN Algorithm

The objective of the algorithm is to predict a position associated with a fingerprint sent by the user. The RSS vector in the fingerprint can be compared to those in the database using the Euclidean distance. We can then find the K nearest neighbors and compute the predicted position using the mean of the positions of the K nearest neighbors.

## The Spoofing Attacks

We can dress up many scenarios of attacks, but here we went for two simple one: the **_Random Attack_** and the **_Targeted Attack_**.

The **_Random Attack_** is the simplest one. The attacker will send fake beacons with random RSS values. The goal is to see how the algorythme will react to this kind of attack. The sender MAC address of the fake beacons is randomly chosen in the list of the MAC addresses of the APs in the database.

The **_Targeted Attack_** is a bit more complex. The attacker will send fake beacons with RSS values that are close to the ones of the APs that are the nearest neighbors of the user. Here, this scenario is more unlikly to happend in real conditions due to signal reflexions... So this case is basically the worst scenario we could ever encounter.

## The Code

To run the simulations, you need to uncomment the line that you want in the **main**.py file. You can run the main script by running the command `python3 .` (make sure your are in the correct directory).

To plot the graphs, you need to uncomment some lines and run the **results_analysis**.py file.

**Version** : Warning, this code is written in Python 3.10, and it will not work with older versions of Python.

## The Documentation

You can find detailed explanations of the algorithm and the results in the [docs](docs) folder.

# Other

**Note**: This repository is still under development.
**_All the data come from the [UJIIndoorLoc](https://www.kaggle.com/datasets/giantuji/UjiIndoorLoc) dataset._**
