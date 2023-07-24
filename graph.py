""" Module to plot radio maps. """
import matplotlib.pyplot as plt
import numpy as np
from radio_map import RadioMap

def plot_radio_map(radio_map: RadioMap, new_figure=False, title=None, args='bx') -> None:
    """ Plot a radio map."""
    positions = []
    for data in radio_map.get_data():
        positions.append(
            (float(data['LONGITUDE']), float(data['LATITUDE'])))
    positions = np.array(positions)
    if new_figure:
        plt.figure()
    plt.plot(positions[:, 0], positions[:, 1], args)
    if title is not None:
        plt.title(title)

def plot_point(point: tuple, args='rx', new_figure=False, title=None, label='') -> None:
    """ Plot a point."""
    if new_figure:
        plt.figure()
    plt.plot(point[0], point[1], args, label=label)
    if label != '':
        plt.legend()
    if title is not None:
        plt.title(title)

def show() -> None:
    """ Show the plot."""
    plt.show()

def plot_confidence_circle(point: tuple, radius: float, color='r') -> None:
    """ Plot a confidence circle."""
    circle = plt.Circle(point, radius, color=color, fill=False)
    plt.gca().add_patch(circle)
