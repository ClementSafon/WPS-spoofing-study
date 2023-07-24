# pylint: skip-file
import sys
import os
import numpy as np
import pytest
import math

current = os.path.dirname(os.path.realpath(__file__))
parent = os.path.dirname(current)
sys.path.append(parent)

from knn_algorithm import run, display

# Mock the RadioMap class
class RadioMap:
    def __init__(self):
        self.rss_data = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
        self.positions = np.array([(1, 1), (2, 2), (3, 3)])
        self.floors = np.array([1, 2, 3])
        self.data = [{'rss': [1, 2, 3], 'LONGITUDE': 1, 'LATITUDE': 1, 'FLOOR': 1}, {'rss': [4, 5, 6], 'LONGITUDE': 2, 'LATITUDE': 2, 'FLOOR': 2}, {'rss': [7, 8, 9], 'LONGITUDE': 3, 'LATITUDE': 3, 'FLOOR': 3}]

    def get_rss(self):
        return self.rss_data

    def get_positions(self):
        return self.positions

    def get_floors(self):
        return self.floors
    
    def get_data(self):
        return self.data

@pytest.fixture
def radio_map():
    return RadioMap()

def test_run(radio_map):
    average_positions, error_positions, average_floors, error_floors = run(2, False, False, radio_map, radio_map, False)
    assert len(average_positions) == 3
    assert len(error_positions) == 3
    assert len(average_floors) == 3
    assert len(error_floors) == 3

def test_run_with_log_penality(radio_map):
    average_positions, error_positions, average_floors, error_floors = run(2, False, False, radio_map, radio_map, True)
    assert len(average_positions) == 3
    assert len(error_positions) == 3
    assert len(average_floors) == 3
    assert len(error_floors) == 3

def test_display(radio_map):
    average_positions, error_positions, average_floors, error_floors = run(2, True, False, radio_map, radio_map, False)
    # Check if display runs without errors (no direct output to check)
    assert True

def test_display_with_empty_average_positions(radio_map):
    average_positions, error_positions, average_floors, error_floors = [], [], [], []
    # Check if display runs without errors (no direct output to check)
    display(radio_map, radio_map, average_positions)
    assert True

def test_print_usage(capsys):
    from knn_algorithm import print_usage
    print_usage()
    captured = capsys.readouterr()
    assert "Usage: python3 knn_algorithm.py -k <k_neigbors> -r <rows, ex: 1,2,5> [--gui] [--verbose]" in captured.out

def test_cli_arguments_parsing():
    args = ['-k', '5', '-r', '1,2,3', '--gui', '--verbose']
    from knn_algorithm import parse_cli_arguments
    k, rows, gui, verbose = parse_cli_arguments(args)
    assert k == 5
    assert rows == [1, 2, 3]
    assert gui is True
    assert verbose is True

def test_cli_arguments_parsing_with_missing_arguments():
    args = []
    from knn_algorithm import parse_cli_arguments
    with pytest.raises(SystemExit):
        parse_cli_arguments(args)

def test_cli_arguments_parsing_with_invalid_k_value():
    args = ['-k', 'abc', '-r', '1,2,3', '--gui', '--verbose']
    from knn_algorithm import parse_cli_arguments
    with pytest.raises(SystemExit):
        parse_cli_arguments(args)

def test_cli_arguments_parsing_with_invalid_r_value():
    args = ['-k', '5', '-r', '1,2,abc', '--gui', '--verbose']
    from knn_algorithm import parse_cli_arguments
    with pytest.raises(SystemExit):
        parse_cli_arguments(args)

def test_cli_arguments_parsing_with_invalid_flag():
    args = ['-k', '5', '-r', '1,2,3', '--unknown_flag', '--verbose']
    from knn_algorithm import parse_cli_arguments
    with pytest.raises(SystemExit):
        parse_cli_arguments(args)

def test_run_average_position():
    from knn_algorithm import run
    target_rss = RadioMap()
    training_rss = RadioMap()
    average_position = run(1, False, False, training_rss, target_rss)[0]
    expected_average_position = np.array([(1, 1), (2, 2), (3, 3)])
    assert np.allclose(average_position, expected_average_position)

def test_knn_algorithm_module():
    from knn_algorithm import RadioMap, run, display, print_usage
    assert True  # Just to ensure the module imports without errors
