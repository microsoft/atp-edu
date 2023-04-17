"""
game interact with monitor and user.
"""
from typing import List, Dict
import pygame

from game_models import SnakeGame, GameAction, Point, GameStatus

raw_resources = {
    'food_image': pygame.image.load('img/food2.png'),
    'snake_image_body': pygame.image.load('img/snakeBody.png'),
    'snake_image_head': pygame.image.load('img/snakeHead.png'),
    'bg_image': pygame.image.load("img/background.png")
    # bg_image = pygame.transform.scale(pygame.image.load("img/background.png"),
    #                                   (self.display_width - block_size, self.display_width - block_size))
}


def init_pygame(display_width, display_height, title='game'):
    pygame.init()
    pygame.font.init()
    pygame.display.set_caption(title)
    return pygame.display.set_mode((display_width, display_height))


def display_single_game(game: SnakeGame, window: pygame.Surface,clean_event=True):
    if clean_event:
        pygame.event.get()  # clean up event queue to avoid screen stuck
    window.fill((255, 255, 255))
    width = window.get_width()
    height = window.get_height()
    _display_single_game_on_specific_area(game, window, Point(0, 0), dis_width=width, dis_height=height,
                                          resources=raw_resources)
    pygame.display.update()


def display_games(games: List[SnakeGame], row, column, window: pygame.Surface, clean_event=True):
    if clean_event:
        pygame.event.get()  # clean up event queue to avoid screen stuck

    width = window.get_width()
    height = window.get_height()
    assert len(games) <=row*column, "游戏数{}大于可最大可显示数{}".format(len(games), row*column)
    window.fill((255, 255, 255))
    width_per_game = int(width/column)
    height_per_game = int(height/row)
    for i,game in enumerate(games):
        current_row = int(i / column)
        current_column = i % column
        top_left = Point(current_column*width_per_game,current_row*height_per_game)
        _display_single_game_on_specific_area(game, window, top_left, dis_width=width_per_game,
                                              dis_height=height_per_game, resources=raw_resources)
    pygame.display.update()


def _display_single_game_on_specific_area(game: SnakeGame, window: pygame.Surface, top_left: Point, *,
                                          dis_width=None, dis_height=None, resources: Dict[str, pygame.Surface] = None,
                                          block_size=None, scaled_resources: Dict[str, pygame.Surface] = None):
    if dis_width and dis_height and resources:
        if dis_height < dis_width + 40:  # 只使用左侧空间如果空间过长
            dis_width = dis_height - 40
        block_size = int(dis_width / (game.game_width + 2))
        assert block_size >= 4, "画面宽度太小，无法绘制"
        dis_width = block_size * (game.game_width + 2)
        dis_height = block_size * (game.game_width + 2) + 40
        scaled_resources = {
            'food_image': pygame.transform.scale(resources['food_image'], (block_size, block_size)),
            'snake_image_body': pygame.transform.scale(resources['snake_image_body'], (block_size, block_size)),
            'snake_image_head': pygame.transform.scale(resources['snake_image_head'], (block_size, block_size)),
            'bg_image': pygame.transform.scale(resources['bg_image'], (dis_width - block_size, dis_width - block_size))
        }

    elif block_size and scaled_resources:
        dis_width = block_size * (game.game_width + 2)
        dis_height = block_size * (game.game_width + 2) + 40
    else:
        raise Exception('Invalid input provided for display')

    player_positions, food_coord = game.get_current_status()
    # display subtitles
    myfont = pygame.font.SysFont('Segoe UI', 15)
    font_bold_big = pygame.font.SysFont('Segoe UI', 15, True)
    text_score = myfont.render('SCORE:', True, (0, 0, 0))
    text_score_number = myfont.render(str(game.round_score), True, (0, 0, 0))
    text_highest = myfont.render('BEST:', True, (0, 0, 0))
    text_highest_number = font_bold_big.render(str(game.record), True, (0, 0, 0))
    window.blit(text_score, (top_left.x+5, top_left.y + dis_height - 40))
    window.blit(text_score_number, (top_left.x+50, top_left.y + dis_height - 40))
    window.blit(text_highest, (top_left.x+5, top_left.y + dis_height - 20))
    window.blit(text_highest_number, (top_left.x+50, top_left.y + dis_height - 20))
    window.blit(scaled_resources['bg_image'], (top_left.x + block_size / 2, top_left.y + block_size / 2))

    # display food first in case the game is on status WIN
    window.blit(scaled_resources['food_image'], (
        top_left.x + (food_coord.x + 1) * block_size, top_left.y + (food_coord.y + 1) * block_size))  # 1 is the boarder

    # display snake
    window.blit(scaled_resources['snake_image_head'],
                (top_left.x + (player_positions[-1].x + 1) * block_size,
                 top_left.y + (player_positions[-1].y + 1) * block_size))  # 1 is the boader size
    for x_temp, y_temp in player_positions[:-1]:
        window.blit(scaled_resources['snake_image_body'],
                    (top_left.x + (x_temp + 1) * block_size, top_left.y + (y_temp + 1) * block_size))

    if game.status is GameStatus.CRASH:
        font_bold_small = pygame.font.SysFont('Segoe UI', 10, True)
        text_failed = font_bold_small.render('You Die!', True, (0, 0, 0))
        window.blit(text_failed, (top_left.x + dis_width / 3, top_left.y + dis_height / 3))
    if game.status is GameStatus.WIN:
        font_bold_big = pygame.font.SysFont('Segoe UI', 20, True)
        text_failed = font_bold_big.render('You WIN!', True, (255, 0, 0))
        window.blit(text_failed, (top_left.x + dis_width / 4, top_left.y + dis_height / 4))


# def _render(self):
#     self._display_ui()
#     self.player.display(self.gameDisplay)
#     self.food.display(self.gameDisplay)


def process_input():
    cmd = None
    events = pygame.event.get()
    for event in events:
        if event.type == pygame.QUIT:
            quit()
        elif event.type == pygame.KEYDOWN:
            if event.key == pygame.K_ESCAPE:
                quit()
            elif event.key == pygame.K_RIGHT:
                cmd = GameAction.RIGHT
            elif event.key == pygame.K_LEFT:
                cmd = GameAction.LEFT
            elif event.key == pygame.K_DOWN:
                cmd = GameAction.DOWN
            elif event.key == pygame.K_UP:
                cmd = GameAction.UP
    return cmd


def calculate_window_size(game_width, row=1, column=1, block_size=10):
    return column*(game_width+2)*block_size, row*((game_width+2)*block_size+40)
