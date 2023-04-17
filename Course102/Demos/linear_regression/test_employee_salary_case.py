import pandas as pd
import numpy as np

import linear_regression_single_variant

if __name__ == "__main__":

    experiences = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
    salaries = np.array([103100, 104900, 106800, 108700, 110400, 112300, 114200, 116100, 117800, 119700, 121600])

    x = experiences
    y = salaries * 0.001 - 100
    linear_regression_single_variant.train_with_visualization(x,y)
