""" Module to plot radio maps. """
import matplotlib.pyplot as plt
import numpy as np
from radio_map import RadioMap

def plot_radio_map(radio_map: RadioMap, new_figure=False, title=None, color="b", marker="x", alpha=1) -> None:
    """ Plot a radio map."""
    pos_2d = radio_map.get_position_matrix()[:, :2]
    if new_figure:
        plt.figure()
    pos_2d = np.unique(pos_2d, axis=0)
    plt.scatter(pos_2d[:, 0], pos_2d[:, 1], color=color, marker=marker, alpha=alpha)
    if title is not None:
        plt.title(title)

def plot_point(point: np.ndarray, args='rx', new_figure=False, title=None, label='') -> None:
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
