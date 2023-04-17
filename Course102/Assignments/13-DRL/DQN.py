import time
from typing import List

import matplotlib.pyplot as plt
import scipy
import seaborn as sns
import pandas as pd
import pygame
import tensorflow as tf
import numpy as np
from tensorflow import keras
from collections import deque
import random, os, sys, logging, datetime, pickle

from pygame_utils import init_pygame, calculate_window_size, display_games
from game_models import SnakeGame, GameStatus, ROUND_FINISH_STATUS, GameAction
from utils import value_abs_error_mean, q_absolute_mean, q_min, q_max, convert_to_observation
from configs import *
import importlib.util

logger = logging.getLogger(__name__)
MODEL_PREDICT_BATCH = 10000  # this should be as big as possible since it doesn't explode memory to speed up training


def dqn_model(state_shape, action_shape, learning_rate):
    init = tf.keras.initializers.HeUniform()
    model = keras.Sequential()
    # ===================== Full connect model
    # model.add(keras.layers.Flatten(input_shape=state_shape, name='flatten'))
    # model.add(keras.layers.Dense(128, activation='relu', kernel_initializer=init, name='l1'))
    # model.add(keras.layers.Dense(64, activation='relu', kernel_initializer=init, name='l2'))
    # model.add(keras.layers.Dense(64, activation='relu', kernel_initializer=init, name='l3'))
    # ==================CNN MODEL
    model.add(keras.layers.Conv2D(32, (1, 1), activation='relu', input_shape=state_shape,kernel_initializer= init))
    # model.add(keras.layers.MaxPooling2D((2, 2)))
    model.add(keras.layers.Conv2D(64, (2, 2), activation='relu',kernel_initializer= init))
    # model.add(keras.layers.MaxPooling2D((2, 2)))
    model.add(keras.layers.Conv2D(64, (3, 3), activation='relu',kernel_initializer= init))
    model.add(keras.layers.Flatten())
    model.add(keras.layers.Dense(128,activation='relu',kernel_initializer=init))

    model.add(keras.layers.Dense(action_shape, activation='linear', kernel_initializer=init))
    model.compile(loss=tf.keras.losses.Huber(), optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
                  metrics=['accuracy', value_abs_error_mean, q_min, q_max, q_absolute_mean])
    return model


def train(replay_memory, model, target_model):
    # todo  训练数据预处理是性能瓶颈，待优化
    if len(replay_memory) < MIN_REPLAY_SIZE:
        return

    mini_batch = random.sample(replay_memory, BATCH_SIZE)
    current_states = np.array([transition[0] for transition in mini_batch])
    current_qs_list = model.predict(current_states,batch_size=MODEL_PREDICT_BATCH)
    new_current_states = np.array([transition[3] for transition in mini_batch])
    future_qs_list = target_model.predict(new_current_states,batch_size=MODEL_PREDICT_BATCH)

    X = []
    Y = []
    for index, (observation, action, reward, new_observation, done) in enumerate(mini_batch):
        if not done:
            max_future_q = reward + DQN_DISCOUNT_FACTOR * np.max(future_qs_list[index])
        else:
            max_future_q = reward

        current_qs = current_qs_list[index]
        # 目标是让当前步的q值等于targetmodel预测的当前值+下一步的最优值
        current_qs[action] = (1 - DQN_LEARNING_RATE) * current_qs[action] + DQN_LEARNING_RATE * max_future_q

        X.append(observation)
        Y.append(current_qs)
    history = model.fit(np.array(X), np.array(Y), batch_size=BATCH_SIZE, verbose=0, shuffle=True)
    logger.info(['{}: {:.5f}'.format(k, v[0]) for k, v in history.history.items()])
    return history.history['loss'][0], history.history['accuracy'][0], history.history['q_min'][0], history.history['q_max'][0], history.history['q_absolute_mean'][0]


def main():
    # todo check ARENA_NUM is the common divisor of STEPS_TO_TRAIN ,STEPS_TO_COPY_MODEL, STEPS_TO_EVALUATE_MODEL and  move corresponding logic out loop
    logger.info("training started")
    logger.info('training with git commit :{}'.format(current_commit))
    logger.info('=================current configs====================')
    logger.info(open('configs.py').read())
    logger.info('=================current configs====================')
    epsilon = 1  # Epsilon-greedy algorithm in initialized at 1 meaning every step is random at the start
    max_epsilon = 1  # You can't explore more than 100% of the time
    min_epsilon = 0.01  # At a minimum, we'll always explore 1% of the time
    initial_epi = 0
    accuracy_records: pd.DataFrame = pd.DataFrame(
        columns=["time", "step", 'avg_score', 'min_score', 'max_score', 'avg_step', 'avg_invalid_step',
                 't_loss', 't_accuracy', 't_q_min', 't_q_max', 't_q_absolute_mean'])

    if DISPLAY_ARENA:
        window_width, window_height = calculate_window_size(ARENA_WIDTH, 3, 5, block_size=20)
        game_window = init_pygame(window_width, window_height)
        pygame_clock = pygame.time.Clock()
    else:
        game_window = None
    positive_count = 0
    negative_count = 0
    zero_count = 0
    games = [SnakeGame(TRAINING_PARAS, arena_width=ARENA_WIDTH) for _ in range(ARENA_NUM)]
    obs_shape = convert_to_observation([(0, 0)], (1, 1)).shape
    main_model = dqn_model(obs_shape, ACTION_SPACE, learning_rate=MODEL_LEARNING_RATE)
    target_model = dqn_model(obs_shape, ACTION_SPACE, learning_rate=MODEL_LEARNING_RATE)
    target_model.set_weights(main_model.get_weights())
    logger.info('===========model summary=========')
    main_model.summary(print_fn=logger.info)

    if SAVE_LOAD_SAMPLES and os.path.isfile(SAMPLE_PATH):
        replay_memory = pickle.load(open(SAMPLE_PATH, 'rb'))
    else:
        replay_memory = deque(maxlen=QUEUE_SIZE)
    current_sample_len = len(replay_memory)

    total_step = 0
    training = False

    observations = [convert_to_observation(*g.get_current_status()) for g in games]
    episode = 0
    while episode < TRAIN_EPISODES:
        rand_num = np.random.rand()
        for i, g in enumerate(games):
            if g.status in ROUND_FINISH_STATUS:
                g.start_new_game()
                episode += 1
                snake_positions, food_position = g.get_current_status()
                observations[i] = convert_to_observation(snake_positions, food_position)  # 对于重新开始的game，从env刷新observation

        if rand_num < epsilon:
            actions = [np.random.choice([0, 1, 2, 3]) for i in range(ARENA_NUM)]
        else:
            actions = np.argmax(main_model.predict(np.array(observations),batch_size=MODEL_PREDICT_BATCH), axis=1)
        new_obs = []
        for i, g in enumerate(games):
            total_step += 1
            action = actions[i]
            observation = observations[i]
            status, step_reward, n_snake_positions, n_food_position, is_valid_command, eaten_food = g.run_one_step(
                action,trainning_model=True)
            new_observation = convert_to_observation(n_snake_positions, n_food_position)
            if status in ROUND_FINISH_STATUS:
                new_obs_meaningless = True
            else:
                new_obs_meaningless = False

            if BALANCE_SAMPLING:
                rand_num = np.random.random()
                if not is_valid_command and rand_num < 0.06:  # observation for invalid command is meaningless
                    replay_memory.append(
                        [observation, action, step_reward, new_observation,
                         new_obs_meaningless])  # treat invalid as a failed step
                    negative_count += 1
                elif eaten_food:
                    replay_memory.append([observation, action, step_reward, new_observation, new_obs_meaningless])
                    positive_count += 1
                elif step_reward < 0:
                    if rand_num < 0.1:  # 10% possibility
                        replay_memory.append([observation, action, step_reward, new_observation, new_obs_meaningless])
                        negative_count += 1
                else:
                    if rand_num < 0.01:  # 1% possibility
                        replay_memory.append([observation, action, step_reward, new_observation, new_obs_meaningless])
                        zero_count += 1
            else:
                if eaten_food:
                    replay_memory.append([observation, action, step_reward, new_observation, new_obs_meaningless])
                    positive_count += 1
                elif step_reward < 0:
                    replay_memory.append([observation, action, step_reward, new_observation, new_obs_meaningless])
                    negative_count += 1
                else:
                    replay_memory.append([observation, action, step_reward, new_observation, new_obs_meaningless])
                    zero_count += 1

            new_obs.append(new_observation)

            if total_step % STEPS_TO_TRAIN == 0:
                train_result = train(replay_memory, main_model, target_model)
                if train_result:
                    t_loss, t_accuracy, t_q_min, t_q_max, t_q_absolute_mean= train_result
                if total_step % STEPS_TO_COPY_MODEL == 0:
                    logger.info('Copying main network weights to the target network weights')
                    target_model.set_weights(main_model.get_weights())
                    logger.info('For round={} , epsilon:{},total_step:{}, memory_length: {}'.format(
                        episode, epsilon, total_step,
                        len(replay_memory)))
                    logger.info(
                        'positive: {}  negative : {} zero: {}'.format(positive_count, negative_count, zero_count))
                training = True

            if total_step % STEPS_TO_EVALUATE_MODEL == 0:
                test_matrics = test_model(target_model, times=500, concurrency=100, window=game_window)
                logger.info("!!!!!!!!!!!!!Test score for step {} is {}".format(total_step, test_matrics))
                accuracy_records = accuracy_records.append({"time": datetime.datetime.now(),
                                                          "step": total_step,
                                                          'avg_score': test_matrics[0],
                                                          'min_score': test_matrics[1],
                                                          'max_score': test_matrics[2],
                                                          'avg_step': test_matrics[3],
                                                          'avg_invalid_step': test_matrics[4],
                                                          'starving_count': test_matrics[5],
                                                          't_loss':t_loss,
                                                          't_accuracy':t_accuracy,
                                                          't_q_min':t_q_min,
                                                          't_q_max':t_q_max,
                                                          't_q_absolute_mean':t_q_absolute_mean}, ignore_index=True)
                sns.lineplot(data=accuracy_records, x='step', y='avg_score', markers=True, )
                plt.pause(0.01)
                if SAVE_MODEL_ETC:
                    plt.savefig('logs/DQN-{}-avg-score.png'.format(dts))
                    accuracy_records.to_csv('logs/DQN-{}-indicators.csv'.format(dts), index=False)

            if SAVE_MODEL_ETC and total_step % STEPS_TO_SAVE_MODEL == 0:
                target_model.save_weights("models/{}/{}.h5".format(dts, total_step))
                target_model.save("models/{}/{}m.h5".format(dts, total_step))

            if SAVE_LOAD_SAMPLES and len(replay_memory) % 20000 == 19999:
                if current_sample_len != len(replay_memory):
                    logger.info("save queue with length:{}".format(len(replay_memory)))
                    pickle.dump(replay_memory, open(SAMPLE_PATH, 'wb'))
                    current_sample_len = len(replay_memory)
            if len(replay_memory) == MIN_REPLAY_SIZE:
                initial_epi = episode

            if status and training:
                logger.debug('For round={} , epsilon:{},total_step:{}, memory_length: {}'.format(
                    episode, epsilon, total_step,
                    len(replay_memory)))
                logger.debug('positive: {}  negative : {} zero: {}'.format(positive_count, negative_count, zero_count))
                training = False

            # if g.round_step-g.last_eaten_step>100:
            #     logger.warning("Game crash caused by hunger.")
            #     g.status = GameStatus.CRASH

        if DISPLAY_ARENA:
            display_games(games[:15], row=3, column=5, window=game_window)
        observations = new_obs

        epsilon = min_epsilon + (max_epsilon - min_epsilon) * np.exp(-EPSILON_DECAY * (episode - initial_epi))

        if len(replay_memory) < MIN_REPLAY_SIZE:
            epsilon = 1


def test_model(model, times=100, starving_step=80, concurrency=100, window=None,investigation_mode=False, interactive_mode=None):
    total_score = 0
    total_round_step = 0
    total_invalid_step = 0
    min_score = 999
    max_score = 0
    starving_count = 0
    games:List[SnakeGame] = [SnakeGame(paras=None, arena_width=ARENA_WIDTH) for _ in range(concurrency)]
    episode = 0

    while episode < times:
        observations = []
        for i, g in enumerate(games):
            starving = (g.round_step-g.last_eaten_step) >= starving_step
            if g.status in ROUND_FINISH_STATUS or starving:
                if starving:
                    starving_count += 1
                episode += 1
                total_score += g.round_score
                total_round_step += g.round_step
                total_invalid_step += g.invalid_step
                min_score = g.round_score if g.round_score < min_score else min_score
                max_score = g.round_score if g.round_score > max_score else max_score
                if investigation_mode:
                    logger.info("Snake died, its latest direction is {}".format(g.snake_direction))
                    logger.info("Total step is {}, score is {}".format(g.round_step,g.round_score))
                g.start_new_game()

                if episode == times:
                    break
            snake_positions, food_position = g.get_current_status()
            observations.append(convert_to_observation(snake_positions, food_position))  # 从env获取observation
        if episode == times:
            continue
        predictions = model.predict(np.array(observations),batch_size=MODEL_PREDICT_BATCH)
        actions = np.argmax(predictions, axis=1)
        if investigation_mode and interactive_mode:
            input()

        for i, g in enumerate(games):
            action = actions[i]
            _, step_score, _, _, is_valid_command, eaten_food = g.run_one_step(action)
            if investigation_mode :
                logger.info("prediction is {},action is{}".format(predictions[i], GameAction(action)))
                if not is_valid_command:
                    logger.info('got a invalid action :{}'.format(action))

        if window:
            display_games(games[:1], row=1, column=1, window=window)
    avg_score = total_score / times
    avg_step = total_round_step / times
    avg_invalid_step = total_invalid_step / times
    return avg_score, min_score, max_score, avg_step, avg_invalid_step, starving_count


if __name__ == '__main__':
    gpus = tf.config.list_physical_devices('GPU')
    if len(gpus)>1:
        tf.config.set_visible_devices(gpus[1], 'GPU')
    current_commit = 'not available'
    if importlib.util.find_spec('git'):
        import git

        try:
            repo = git.Repo(search_parent_directories=True)
            current_commit = repo.head.object.hexsha
        except Exception:
            pass

    dts = datetime.datetime.now().strftime('%m%d%H%M%S')

    sh = logging.StreamHandler(sys.stdout)
    handlers = [sh]
    if SAVE_MODEL_ETC:
        fh = logging.FileHandler('logs/DQN-{}.log'.format(dts), encoding='utf-8')
        os.mkdir('models/{}'.format(dts))
        handlers.append(fh)
    # noinspection PyArgumentList
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s', handlers=handlers)
    tf.random.set_seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)
    random.seed(RANDOM_SEED)
    main()
