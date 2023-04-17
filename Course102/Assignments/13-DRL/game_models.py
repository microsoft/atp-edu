from collections import namedtuple
from enum import Enum
from typing import List
import random
import numpy as np


class GameAction(Enum):
    UP = 0
    DOWN = 1
    LEFT = 2
    RIGHT = 3


class GameStatus(Enum):
    INITIAL = 0
    CRASH = 1
    WIN = 2


ROUND_FINISH_STATUS = [GameStatus.CRASH, GameStatus.WIN]
Point = namedtuple("Point", ['x', 'y'])


class SnakeGame:
    """
    这里的坐标为x轴从左到右，y轴从上到下（与通常上下相反）
    """

    def __init__(self, paras=None, arena_width=20):
        self.paras = paras
        self.game_width = arena_width
        self.game_height = arena_width
        self.record = 0
        self.status = GameStatus.INITIAL
        self.player: Player = Player(self)
        self.food = Food(self)
        self.round_score = 0
        self.round_step = 0
        self.last_eaten_step = 0
        self.invalid_step = 0
        self.snake_direction = GameAction.RIGHT

    def start_new_game(self):
        self.status = GameStatus.INITIAL
        self.player = Player(self)
        self.food = Food(self)
        self.round_score = 0
        self.round_step = 0
        self.last_eaten_step = 0
        self.invalid_step = 0
        self.snake_direction = GameAction.RIGHT

    def run_one_step(self, command,trainning_model=False):
        """
        用于模型训练
        必然会前进一步，如果command无效（向后），则按当前方向行走，此情况下后续状态（和command为无关）无法用于训练
        """
        step_score = 0
        eaten_food = False
        if command == -1:
            command = np.random.choice(GameAction)
        else:
            command = GameAction(command)
        is_valid_command = self.is_valid_command(command)
        if is_valid_command:
            self.snake_direction = command
            eaten_food = self.check_eat_food_and_update()
            if eaten_food:
                step_score += 1
            if self.status == GameStatus.CRASH:
                step_score -= 5
        else:
            self.invalid_step += 1
            if not trainning_model:
                self.check_eat_food_and_update()
            step_score -= 5
        return self.status, step_score, self.player.positions, self.food.coord, is_valid_command, eaten_food

    def check_eat_food_and_update(self):
        self.round_step += 1
        eaten_food = False
        if self._will_eat_food():
            eaten_food = True
            self.player.do_eat(self.snake_direction)
            self.round_score += 1
            self.last_eaten_step= self.round_step
            if self.player.size == self.game_width * self.game_height:
                self.status = GameStatus.WIN
            else:
                self.food = Food(self)
        elif self._will_die():
            self.status = GameStatus.CRASH
        else:
            self.player.do_move(self.snake_direction)
        if self.status in ROUND_FINISH_STATUS:
            self.record = self.round_score if self.round_score > self.record else self.record
        return eaten_food

    def is_valid_command(self, command):
        if command is None:
            return False
        if self.player.size == 1:
            return True
        next_head = SnakeGame.get_next_position(command, self.player.head)
        if next_head == self.player.positions[-2]:
            return False
        else:
            return True

    def _will_eat_food(self):
        next_head = SnakeGame.get_next_position(self.snake_direction, self.player.head)
        if next_head == self.food.coord:
            return True
        else:
            return False

    @classmethod
    def get_next_position(cls, direction, p) -> Point:
        if direction == GameAction.LEFT:
            return Point(p.x - 1, p.y)
        elif direction == GameAction.UP:
            return Point(p.x, p.y - 1)
        elif direction == GameAction.RIGHT:
            return Point(p.x + 1, p.y)
        elif direction == GameAction.DOWN:
            return Point(p.x, p.y + 1)



    def _will_die(self):
        next_head = SnakeGame.get_next_position(self.snake_direction, self.player.head)
        if next_head.x < 0 or next_head.x >= self.game_width \
                or next_head.y < 0 or next_head.y >= self.game_height \
                or next_head in self.player.positions[1:]:  # 是否和除尾巴之外的点有重叠
            return True
        else:
            return False

    def get_current_status(self):
        return self.player.positions, self.food.coord


class Player(object):
    def __init__(self, game: SnakeGame):
        self.head = Point(int(0.5 * game.game_width), int(0.5 * game.game_height))
        self.positions: List[Point] = []  # last point in list is head
        self.positions.append(self.head)
        self.size = 1
        if self.head.x - 1 >= 0:
            self.positions.insert(0, Point(self.head.x - 1, self.head.y))
            self.size += 1
        if self.head.x - 2 >= 0:
            self.positions.insert(0, Point(self.head.x - 2, self.head.y))
            self.size += 1

    def update_position(self, next_head):
        assert abs(next_head.x - self.positions[-1].x) + abs(
            next_head.y - self.positions[-1].y) == 1, '新位置与旧位置距离不为1，current_head:{},next_head:{}'.format(
            self.positions[-1], next_head)

        if self.size > 1:
            for i in range(0, self.size - 1):
                self.positions[i] = self.positions[i + 1]
        self.positions[-1] = next_head

    def do_move(self, action):
        self.head = SnakeGame.get_next_position(action, self.head)
        self.update_position(self.head)

    def do_eat(self, action):
        self.head = SnakeGame.get_next_position(action, self.head)
        self.positions.append(self.head)
        self.size += 1


class Food(object):
    def __init__(self, game: SnakeGame):
        # self.image = game.food_image
        self.coord: Point = self.get_food_coord(game)
        # self.block_size = game.block_size

    @staticmethod
    def get_food_coord(game: SnakeGame):
        candidates = []
        for c_x in range(game.game_width):  # include 0 and width-1
            for c_y in range(game.game_height):
                point = Point(c_x, c_y)
                if point not in game.player.positions:
                    candidates.append(point)
        if not candidates:
            print(game.player.positions)
        return random.choice(candidates)
