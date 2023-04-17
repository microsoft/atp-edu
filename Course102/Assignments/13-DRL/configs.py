RANDOM_SEED = 5
ARENA_WIDTH = 20  # the width of snake arena, height is the same
ACTION_SPACE = 4
ARENA_NUM = 500   # arenas to play in parallel

# An episode is a full game
TRAIN_EPISODES = 2000000
EPSILON_DECAY = 0.0002
MIN_REPLAY_SIZE = 10000

# ARENA CONFIGS
DISPLAY_ARENA = True
SAVE_MODEL_ETC = True  # if we need to save model logs and so on

# TRAINING CONFIGS
MODEL_LEARNING_RATE = 0.001
DQN_LEARNING_RATE = 0.7  # Learning rate
DQN_DISCOUNT_FACTOR = 0.99
BATCH_SIZE = 64 * 128


STEPS_TO_TRAIN = 1000
STEPS_TO_COPY_MODEL = 20000
STEPS_TO_EVALUATE_MODEL = 100000
QUEUE_SIZE = 200_000
STEPS_TO_SAVE_MODEL = 2_000_000

# Not quite useful
BALANCE_SAMPLING = False

SAVE_LOAD_SAMPLES = False
SAMPLE_PATH = 'initial_55_mini_reward_memoryttt.pkl'
TRAINING_PARAS={
    # currently nothing is needed
}