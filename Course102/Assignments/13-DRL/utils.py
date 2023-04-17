import tensorflow.keras.backend as K
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from configs import ARENA_WIDTH


def accuracy1(y_true, y_pred):
    """This function is the same with keras multi-category accuracy"""
    return K.mean(K.equal(K.argmax(y_true, axis=-1), K.argmax(y_pred, axis=-1)))


def value_abs_error_mean(y_true, y_pred):
    return K.mean(K.abs(y_true - y_pred))


def q_absolute_mean(y_true, y_pred):
    return K.mean(K.abs(y_pred))


def q_min(y_true, y_pred):
    return K.min(y_pred)


def q_max(y_true, y_pred):
    return K.max(y_pred)


def display_observation(observation: np.ndarray):
    ob = observation.copy()
    ob = ob.astype(np.int)
    ob[:, :, 1] = ob[:, :, 1] * 4
    ob[:, :, 2] = ob[:, :, 2] * 8
    ob = np.sum(ob, axis=2).T  # transpose 让x为横轴，y为纵轴
    for i in range(ob.shape[0]):
        print(' '.join(map(str, ob[i])))  # up 会导致y-=1  所以这里不用处理


def plot_csv(path):
    df = pd.read_csv(path)
    sns.pairplot(df, x_vars=['step'], y_vars=['avg_score', 'min_score', 'max_score', 'avg_step','avg_invalid_step'])
    plt.show()


def convert_to_observation(snake_poses, food_pos):
    """
    1 is body
    2 is head
    3 is food
    0 is empty
    """
    state = np.zeros((ARENA_WIDTH, ARENA_WIDTH, 3), dtype=np.float32)

    for x, y in snake_poses[:-1]:  # 0层身体，1层头 2层food
        state[x][y][0] = 1
    state[snake_poses[-1][0]][snake_poses[-1][1]][1] = 1
    state[food_pos[0]][food_pos[1]][2] = 1
    return state


    #不可用方案
    # state = np.zeros((ARENA_WIDTH,ARENA_WIDTH),dtype=np.float32)
    # for x,y in snake_poses[:-1]:
    #     state[x][y] = 128
    # state[snake_poses[-1][0]][snake_poses[-1][1]] = 64
    # state[food_pos[0]][food_pos[1]] = 255
    # return state

    #可用，但是收敛更慢，上限也低。
    # state = np.zeros((ARENA_WIDTH,ARENA_WIDTH),dtype=np.float32)
    # for x,y in snake_poses[:-1]:
    #     state[x][y] = 1
    # state[snake_poses[-1][0]][snake_poses[-1][1]] = 2
    # state[food_pos[0]][food_pos[1]] = 3
    # return state