# pylint: skip-file

import sys
import os
import tempfile
import pytest
import numpy as np

current = os.path.dirname(os.path.realpath(__file__))
parent = os.path.dirname(current)
sys.path.append(parent)

from radio_map import RadioMap


@pytest.fixture
def radio_map():
    return RadioMap()


def test_load_existing_file(radio_map):
    with tempfile.NamedTemporaryFile(mode='w', delete=False) as temp_file:
        temp_file.write(
            "LONGITUDE,LATITUDE,FLOOR,BUILDINGID,SPACEID,RELATIVEPOSITION,USERID,PHONEID,TIMESTAMP,WAP1,WAP2\n")
        temp_file.write(
            "1.234,5.678,1,100,200,300,1,123456,2023-07-24 12:34:56,-70,-65\n")
    radio_map.load(temp_file.name)
    assert len(radio_map.get_data()) == 1


def test_load_non_existing_file(radio_map):
    with pytest.raises(FileExistsError):
        radio_map.load("non_existing_file.csv")


def test_load_invalid_csv(radio_map):
    with tempfile.NamedTemporaryFile(mode='w', delete=False) as temp_file:
        temp_file.write("INVALID_HEADER\n")
        temp_file.write(
            "1.234,5.678,1,100,200,300,1,123456,2023-07-24 12:34:56,-70,-65\n")
    with pytest.raises(ValueError):
        radio_map.load(temp_file.name)


def test_fork_valid_rows(radio_map):
    radio_map.load("tests/example.csv")
    forked_map = radio_map.fork([0, 2, 4])
    assert len(forked_map.get_data()) == 3


def test_fork_invalid_rows(radio_map):
    radio_map.load("tests/example.csv")
    with pytest.raises(ValueError):
        radio_map.fork([0, 10, 20])


def test_spoof_valid_rows(radio_map):
    radio_map.load("tests/example.csv")
    spoofing_map = RadioMap()
    spoofing_map.load("tests/example.csv")
    radio_map.spoof(0, spoofing_map, 1)
    assert (radio_map.get_data()[0]['rss'] == np.array([-80, -70, -30, -70, -70, -80, 100, 100, 100, 100])).any()


def test_spoof_invalid_rows(radio_map):
    radio_map.load("tests/example.csv")
    spoofing_map = RadioMap()
    spoofing_map.load("tests/example.csv")
    with pytest.raises(ValueError):
        radio_map.spoof(0, spoofing_map, 10)


def test_random_spoof_valid_row(radio_map):
    radio_map.load("tests/example.csv")
    radio_map.random_spoof(0, seed=123)
    assert len(radio_map.get_data()[0]['rss']) == 10


def test_random_spoof_invalid_row(radio_map):
    radio_map.load("tests/example.csv")
    with pytest.raises(ValueError):
        radio_map.random_spoof(10, seed=123)


def test_get_data(radio_map):
    radio_map.load("tests/example.csv")
    assert len(radio_map.get_data()) == 9


def test_get_data_by_row(radio_map):
    radio_map.load("tests/example.csv")
    data = radio_map.get_data_by_row([0, 2, 4])
    assert len(data) == 3


def test_get_rss(radio_map):
    radio_map.load("tests/example.csv")
    rss_data = radio_map.get_rss()
    assert rss_data.shape[0] == len(radio_map.get_data())


def test_get_positions(radio_map):
    radio_map.load("tests/example.csv")
    positions = radio_map.get_positions()
    assert positions.shape[0] == len(radio_map.get_data())


def test_get_floors(radio_map):
    radio_map.load("tests/example.csv")
    floors = radio_map.get_floors()
    assert floors.shape[0] == len(radio_map.get_data())
