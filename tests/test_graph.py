# pylint: skip-file
import sys
import os
import pytest
import numpy as np
import matplotlib.pyplot as plt

current = os.path.dirname(os.path.realpath(__file__))
parent = os.path.dirname(current)
sys.path.append(parent)

from radio_map import RadioMap
from graph import plot_radio_map, plot_point, show, plot_confidence_circle


@pytest.fixture
def radio_map():
    radio_map = RadioMap()
    radio_map.data = [
        {'LONGITUDE': 1.234, 'LATITUDE': 5.678},
        {'LONGITUDE': 2.345, 'LATITUDE': 6.789},
        {'LONGITUDE': 3.456, 'LATITUDE': 7.890},
    ]
    return radio_map


def test_plot_radio_map(radio_map):
    # Call plot_radio_map and check if it runs without errors
    plot_radio_map(radio_map)
    # If no exceptions are raised, the test passes


def test_plot_radio_map_with_title(radio_map):
    # Call plot_radio_map with title and check if it runs without errors
    plot_radio_map(radio_map, title="Radio Map Plot")
    # If no exceptions are raised, the test passes


def test_plot_point():
    # Call plot_point and check if it runs without errors
    plot_point((1.234, 5.678))
    # If no exceptions are raised, the test passes


def test_plot_point_with_title():
    # Call plot_point with title and check if it runs without errors
    plot_point((1.234, 5.678), title="Point Plot")
    # If no exceptions are raised, the test passes


def test_plot_confidence_circle():
    # Call plot_confidence_circle and check if it runs without errors
    plot_confidence_circle((1.234, 5.678), radius=0.5)
    # If no exceptions are raised, the test passes


def test_show():
    # Call show and check if it runs without errors
    show()
    # If no exceptions are raised, the test passes


def test_show_after_plot_radio_map(radio_map):
    # Call plot_radio_map and check if it runs without errors
    plot_radio_map(radio_map)
    # Call show and check if it runs without errors
    show()
    # If no exceptions are raised, the test passes


def test_show_after_plot_point():
    # Call plot_point and check if it runs without errors
    plot_point((1.234, 5.678))
    # Call show and check if it runs without errors
    show()
    # If no exceptions are raised, the test passes
