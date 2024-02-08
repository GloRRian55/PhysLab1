from typing import Dict, Union, List, Tuple, Optional

import numpy as np
from matplotlib import pyplot
from random import gauss

import warnings
warnings.simplefilter('ignore', np.RankWarning)

INT_OR_FLOAT = Union[int, float]


def generate_data(number: int, mean: float,
                  sigma: float, accuracy: Optional[int] = 100
                  ) -> Dict[float, int]:
    """
    Function for generate random data map by Gaussian random function

    :param number: Number of random generation
    :param mean: Average value of Gaussian random generation
    :param sigma: The standard deviation of Gaussian random generation
    :param accuracy: Precision of generated numbers in integer
    format (e.g 100 - precision to hundredths); default - 100
    :return: Sorted by keys map whose keys is the set of
    the generated numbers and values is respectively a number of their repetitions
    """
    rnd_dict: Dict[float, int] = dict()
    for i in range(number):
        sample: float = round(gauss(mean, sigma), len(str(accuracy)))
        if sample in rnd_dict.keys():
            rnd_dict[sample] += 1
        else:
            rnd_dict[sample] = 1
    return dict(sorted(rnd_dict.items()))


def approximate_plot(x_val: List[INT_OR_FLOAT], y_val: List[INT_OR_FLOAT]
                     ) -> np.ndarray[float]:
    """
    Approximation function for a graph using arrays of values ​​of its axes

    :param x_val: Array of x-axis values of initial plot
    :param y_val: Array of y-axis values of initial plot
    :return: Array of y-axis numbers after approximation
    """
    theta: np.array = np.polyfit(np.array(x_val), np.array(y_val), deg=100)
    model: np.poly1d = np.poly1d(theta)
    return model(x_val)


if __name__ == '__main__':
    SAMPLES_NUMBER: int = 1000
    SAMPLES_ACCURACY: int = 100
    SAMPLES_MEAN: float = 5
    SAMPLES_SIGMA: float = 0.01
    # For a beautiful result, it is required that
    # SAMPLES_SIGMA * SAMPLES_ACCURACY = 1

    x_values = list()
    y_values = list()

    random_dict = generate_data(SAMPLES_NUMBER, SAMPLES_MEAN, SAMPLES_SIGMA, SAMPLES_ACCURACY)
    for x, y in random_dict.items():
        x_values.append(x)
        y_values.append(y)

    y_app_values = approximate_plot(x_values, y_values)

    pyplot.title("Распределение по Гауссу")
    pyplot.xlabel("Значения времени, c")
    pyplot.ylabel("Количество, штук")

    pyplot.plot(x_values, y_values, "ob")
    pyplot.plot(x_values, y_app_values, )
    pyplot.show()
