## Scenario Description

The spoofing attack in positioning services is a problem that can introduce imprecision and even lead the user to believe they are in a location they are not. Attackers send fake AP beacons that affect the fingerprint sent by the user to the server.

To analyze the performance of our algorithms, we will use the normal validation dataset and create some corrupted versions.

We will simulate two attack scenarios. In the first scenario, the attacker knows some of the MAC addresses of APs in the Radio Map. They will emit beacons with those MAC addresses but will not choose specific MAC addresses. These addresses are randomly selected from the list of MAC addresses that the attacker knows. This is a basic attack that aims to create imprecision in positioning.

The second scenario is more severe. The attacker sends beacons with MAC addresses that are in the same area with specific RSSI values. The attacker aims to make the user believe they are in a specific place (where the fake APs are normally located). This is the worst case. In the real life, because of reflections, the attacker will have fewer chances to have corresponding RSSI values for the victim. **However, in all scenarios, the user still detects real beacons from real APs. In some other situations, worst-case scenarios can be found where the user only receives fake beacons.**

## Creation of the corrupted datasets

As the attacker can send many beacons from many APs, we will generate cases where the attack spoofs from 1 to 10 APs. For each case, we will generate 10 corrupted datasets.

In the first scenario, we will simply add n random RSSI values to each fingerprint in the Validation dataset. The RSSI values are randomly chosen between -70 and -30.

In the second scenario, to corrupt a fingerprint from the validation dataset, we will take another fingerprint from the validation dataset that is in another building (sufficiently far away to attempt to displace the user). Then, we will copy n values from the second fingerprint into the corrupted one (in the case of two RSSI values not equal to 100, we will take the higher one).

We now have 10 datasets of corrupted fingerprints for each scenario. We can proceed to evaluate the performance of the algorithms on these datasets.
