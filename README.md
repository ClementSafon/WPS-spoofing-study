## WPS - Study on WiFi based Spoofing attacks

### Description

This repository contains KNN positioning algorythme based on WiFi fingerprints. It is used to simulate localization requests and Spoofing attacks.

In terme of positioning services, sometimes, GPS is not precise enought to give to the user a proper response. That is why, WiFi based positioning services are commonly used in indoor localization systems. It is based on the RSS (Received Signal Strength) of beacons emited by WiFi access points.

However, beacons are not secure and can be spoofed by an attacker. Indeed an attacker can send fake beacons, as if they were comming from Access Point (AP) that are not really here. The user could then be localized in a wrong position.

In this repository, we will try to implement some KNN algorythme and to evaluate their performances in terms of accuracy and robustness against spoofing attacks.

### The KNN Algorithm

The objective of the algorythme is to find a predicted position associated with a fingerprint sent by the user. The RSS Vector in the fingerprint can be compared to the ones in the database using the Euclidean distance. Then we can find the K nearest neighbors and compute the predicted position using the mean of the positions of the K nearest neighbors.

### The Spoofing Attacks

We can dress up many scenarios of attacks, but here we went for two simple one: the **_Random Attack_** and the **_Targeted Attack_**.

The **_Random Attack_** is the simplest one. The attacker will send fake beacons with random RSS values. The goal is to see how the algorythme will react to this kind of attack. The sender MAC address of the fake beacons is randomly chosen in the list of the MAC addresses of the APs in the database.

The **_Targeted Attack_** is a bit more complex. The attacker will send fake beacons with RSS values that are close to the ones of the APs that are the nearest neighbors of the user. Here, this scenario is more unlikly to happend in there condition due to signal reflexions... So this case is basically the worst scenario we could ever encounter.

### The Results

To implement the KNN algorythme, the first thing is to compute the euclidian distances between the RSS vectors of the fingerprints in the database and the RSS vector of the fingerprint sent by the user. But since the RSS values are not equally distributed in the two RSS vectors, we need to choose how to compute the euclidian distance.
For example, we need to compare two RSS vectors like these ones:

````[100,100,-50,-76,100]
[100,-72,-78,-86,100]```

We can see that the second value is a no value for the first and a correct RSSI value for the second.

At this point, we have to filter the data from the database, that we will compare, and then maybe adjuste the calculation of the euclidian distance.

So we can set a threshold on the RSSI values that are in common between the two vector, we will call this method the Shared Coordinates one (SC). But we can also set a limit on the number of "Unshared Coordinates" the UC methode. And finally we can have a "Variable Threashol" (VT) that depend of the ratio between the number of shared coordinates and the number of unshared coordinates.

After that, to compute the euclidian distance, we can chose either to add 0 for the calculation between unshared coordinates, or to had something proportionnal to the number of unshared coordinates for example.

### Method Accuracy

First thing to evaluate is the accuracy of the method. To do so, we will use the UJIIndoorLoc dataset. We will use the training dataset to build the database and the validation to test the algorythme.

To do so, we must found the best combination of Neighbors number (K) and threashold limit (L) for the filtration of the data.
For each method we have a 3D graph that can give us the lowest mean error for each combination of K and L.

For example, here is the graph for the SC method:

![SC]()

We can see that the best combination is K=11 and L=7. So we will use this combination for the rest of the study.

We can do the same for the two other methods:

![UC]()
![VT]()

And we can see that the best combination for the UC method is K=7 and L=16 and for the VT method is K=11 and L=0.64.

### Create corrupted datasets

In order to test the robustness of the algorythme against spoofing attacks, we need to create corrupted datasets. To do so, we will use the validation dataset of the UJIIndoorLoc dataset.

- For the first scenario, we will create 10 copy of the validation dataset, and add from 1 to 10 random RSSI values to all the fingerprints in the dataset. It will allow us to compare the robustness of the algorythme in function of the number of spoofed access points.

- For the second scenario, we will do the same, but instead of adding some random RSSI values, we will copy values from the validation dataset, were the fingeprint is located in an other building (like if the attacker would like to change to vicitm's position by at least one building).

### Basic Positioning Algorythme Robustness

Now that we have our corrupted datasets, we can test the robustness of the algorythme against spoofing attacks. To do so, we will compare the result between the corrupted fingerprint, the original fingerprint and the actual position of the fingerprint.

We can take several parameters to see the robusteness of our algorythme. First the number of successful attacks, that mean that the position estimation between the original fingerprint and the corrupted one is different, and the position estimation of the corrupted one is not null. Then we can take the mean error between the original fingerprint and the corrupted one, and the mean error between the corrupted fingerprint and the actual position. Finally, we can also take the number of KNN fail with the corrupted fingerprints. Indeed, because of the threshold, and the possible unusual RSSI values in the fingerprint, the filtered fingerprints that will serve to compute euclidian distances can be empty. In this case, the algorythme will fail to find the K nearest neighbors and will return a null position.


**Note**: This repository is still under development.
**_All the fingerprints data come from the [UJIIndoorLoc](https://www.kaggle.com/datasets/giantuji/UjiIndoorLoc) dataset._**
````
