from RadioMap import *
import matplotlib.pyplot as plt


class Map():

    def __init__(self) -> None:
        pass

    def plot_radioMap(self, radioMap: RadioMap, new_figure=False, title=None, args='bx') -> None:
        positions = []
        for data in radioMap.get_data():
            positions.append(
                (float(data['LONGITUDE']), float(data['LATITUDE'])))
        positions = np.array(positions)
        if new_figure:
            plt.figure()
        plt.plot(positions[:, 0], positions[:, 1], args)
        if title is not None:
            plt.title(title)

    def plot_point(self, point: tuple, args='rx', new_figure=False, title=None, label='') -> None:
        if new_figure:
            plt.figure()
        plt.plot(point[0], point[1], args, label=label)
        if label != '':
            plt.legend()
        if title is not None:
            plt.title(title)

    def show(self) -> None:
        plt.show()

    def plot_confidence_circle(self, point: tuple, radius: float, args='r--') -> None:
        circle = plt.Circle(point, radius, color='r', fill=False)
        plt.gca().add_patch(circle)
