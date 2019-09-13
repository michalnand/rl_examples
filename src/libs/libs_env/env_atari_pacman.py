import libs.libs_env.env_atari_interface as env_atari_interface
import libs.libs_gl_gui.gl_gui as libs_gl_gui

import numpy
import time
import random

class EnvAtariPacman(env_atari_interface.EnvAtariInterface):

    def __init__(self, size = 32):
        env_atari_interface.EnvAtariInterface.__init__(self, size)

        self.actions_count  = 4

        self.game_height = (int)(self.height)
        self.game_width  = (int)(self.width)

        #init game
        self.reset()

        self.window_name = "PACMAN"


    def reset(self):
        self.element_size = 3
        self.clear_game_screen()
        self.__init_board()
        self.__respawn()

    def _print(self):
        print("move=", self.get_move(), "  score=", self.get_score(), " game=", self.get_games_count())

    def do_action(self, action):

        self.reward = 0.0
        self.set_no_terminal_state()

        if action == 0:
            if self.board[self.pacman_y][self.pacman_x + 1] != 1:
                self.pacman_x+= 1
        elif action == 1:
            if self.board[self.pacman_y][self.pacman_x - 1] != 1:
                self.pacman_x-= 1
        elif action == 2:
            if self.board[self.pacman_y + 1][self.pacman_x] != 1:
                self.pacman_y+= 1
        elif action == 3:
            if self.board[self.pacman_y - 1][self.pacman_x] != 1:
                self.pacman_y-= 1

        if self.board[self.pacman_y][self.pacman_x] == 2:
            self.board[self.pacman_y][self.pacman_x] = 0
            self.reward = 1.0

        if self.__count_remaining() == 0:
            self.reward = 10.0
            self.set_terminal_state()
            self.next_game()
            self.__respawn()
            self.__init_board()

        self.__process_ghots_a()
        self.__process_ghots_b()

        if self.ghost_a_x == self.pacman_x and self.ghost_a_y == self.pacman_y or self.ghost_b_x == self.pacman_x and self.ghost_b_y == self.pacman_y:
           self.reward = -5.0
           self.__respawn()

        self.clear_game_screen()

        for y in range(0, self.board_size):
            for x in range(0, self.board_size):
                item = self.board[y][x]
                if item == 1:
                    self.__put_wall(x, y)
                elif item == 2:
                    self.__put_food(x, y)

        if (self.get_iterations()+1)%5000 == 0:
            self.set_terminal_state()
            self.reset()
            self.next_game()

        self.__put_ghost_a()
        self.__put_ghost_b()
        self.__put_pacman()

        self.update_state()
        self.next_move()


    def __respawn(self):
        self.pacman_x = len(self.board[0])//2
        self.pacman_y = len(self.board)//2

        self.ghost_a_x = self.pacman_x
        self.ghost_a_y = self.pacman_y - 2
        self.ghots_a_state = 0

        self.ghost_b_x = self.pacman_x
        self.ghost_b_y = self.pacman_y + 2
        self.ghots_b_state = 0

    def __init_board(self):
        self.board = [
                        [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                        [1, 2, 2, 2, 2, 1, 2, 2, 2, 2, 1],
                        [1, 2, 1, 1, 2, 1, 2, 1, 1, 2, 1],
                        [1, 2, 1, 2, 2, 2, 2, 2, 1, 2, 1],
                        [1, 2, 1, 2, 1, 1, 1, 2, 1, 2, 1],
                        [1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 1],
                        [1, 2, 1, 2, 1, 1, 1, 2, 1, 2, 1],
                        [1, 2, 1, 2, 2, 2, 2, 2, 1, 2, 1],
                        [1, 2, 1, 1, 2, 1, 2, 1, 1, 2, 1],
                        [1, 2, 2, 2, 2, 1, 2, 2, 2, 2, 1],
                        [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                    ]
        self.board_size = 11

    def __saturate(self, value, min, max):
        if value > max:
            value = max

        if value < min:
            value = min

        return value

    def __put_wall(self, x, y):
        ox = self.width//4 - self.width//8
        oy = self.height//4 - self.height//8
        color = self.item_to_color(14)
        for y_ in range(0, self.element_size):
            for x_ in range(0, self.element_size):
                self.set_game_screen_element(x*self.element_size + x_ + ox, y*self.element_size + y_ + oy, color)

    def __put_food(self, x, y):
        ox = self.width//4 - self.width//8
        oy = self.height//4 - self.height//8
        color = self.item_to_color(3)
        self.set_game_screen_element(x*self.element_size + self.element_size//2 + ox, y*self.element_size + self.element_size//2 + oy, color)

    def __put_ghost_a(self):
        ox = self.width//4 - self.width//8
        oy = self.height//4 - self.height//8
        x = self.ghost_a_x
        y = self.ghost_a_y
        color = self.item_to_color(1)
        for y_ in range(0, self.element_size):
            for x_ in range(0, self.element_size):
                self.set_game_screen_element(x*self.element_size + x_ + ox, y*self.element_size + y_ + oy, color)

        color = self.item_to_color(0)
        self.set_game_screen_element(x*self.element_size + 1 + ox, y*self.element_size + 1 + oy, color)
        self.set_game_screen_element(x*self.element_size + 1 + ox, y*self.element_size + 2 + oy, color)


    def __put_ghost_b(self):
        ox = self.width//4 - self.width//8
        oy = self.height//4 - self.height//8
        x = self.ghost_b_x
        y = self.ghost_b_y
        color = self.item_to_color(7)
        for y_ in range(0, self.element_size):
            for x_ in range(0, self.element_size):
                self.set_game_screen_element(x*self.element_size + x_ + ox, y*self.element_size + y_ + oy, color)

        color = self.item_to_color(0)
        self.set_game_screen_element(x*self.element_size + 1 + ox, y*self.element_size + 1 + oy, color)
        self.set_game_screen_element(x*self.element_size + 1 + ox, y*self.element_size + 2 + oy, color)


    def __put_pacman(self):
        ox = self.width//4 - self.width//8
        oy = self.height//4 - self.height//8
        x = self.pacman_x
        y = self.pacman_y
        color = self.item_to_color(4)
        for y_ in range(0, self.element_size):
            for x_ in range(0, self.element_size):
                self.set_game_screen_element(x*self.element_size + x_ + ox, y*self.element_size + y_ + oy, color)

        color = self.item_to_color(0)
        self.set_game_screen_element(x*self.element_size + 2 + ox, y*self.element_size + 1 + oy, color)

    def __count_remaining(self):
        result = 0
        for y in range(0, self.board_size):
            for x in range(0, self.board_size):
                if self.board[y][x] == 2:
                    result+= 1
        return result

    def __process_ghots_a(self):
        x = self.ghost_a_x
        y = self.ghost_a_y

        if self.__is_crossroad(x, y) == 4:
            idx = random.randint(0, 3)
            if idx == 0 and self.board[y][x+1] != 1:
                self.ghost_a_x+= 1
            elif idx == 1 and self.board[y][x-1] != 1:
                self.ghost_a_x-= 1
            elif idx == 2 and self.board[y+1][x] != 1:
                self.ghost_a_y+= 1
            elif idx == 3 and self.board[y-1][x] != 1:
                self.ghost_a_y-= 1

        else:

            if self.ghots_a_state == 0:
                if self.board[y][x+1] != 1:
                    self.ghost_a_x+= 1
                elif self.board[y+1][x] != 1:
                    self.ghost_a_y+= 1
                else:
                    self.ghots_a_state = 1
            elif self.ghots_a_state == 1:
                if self.board[y][x-1] != 1:
                    self.ghost_a_x-= 1
                elif self.board[y-1][x] != 1:
                    self.ghost_a_y-= 1
                else:
                    self.ghots_a_state = 0

    def __process_ghots_b(self):
        x = self.ghost_b_x
        y = self.ghost_b_y

        if self.__is_crossroad(x, y) == 4:
            idx = random.randint(0, 3)
            if idx == 0 and self.board[y][x+1] != 1:
                self.ghost_b_x+= 1
            elif idx == 1 and self.board[y][x-1] != 1:
                self.ghost_b_x-= 1
            elif idx == 2 and self.board[y+1][x] != 1:
                self.ghost_b_y+= 1
            elif idx == 3 and self.board[y-1][x] != 1:
                self.ghost_b_y-= 1
        else:

            if self.ghots_b_state == 0:
                if self.board[y][x-1] != 1:
                    self.ghost_b_x-= 1
                elif self.board[y+1][x] != 1:
                    self.ghost_b_y+= 1
                else:
                    self.ghots_b_state = 1

            elif self.ghots_b_state == 1:
                if self.board[y][x+1] != 1:
                    self.ghost_b_x+= 1
                elif self.board[y-1][x] != 1:
                    self.ghost_b_y-= 1
                else:
                    self.ghots_b_state = 0

    def __is_crossroad(self, x, y):
        count = 0
        if self.board[y][x+1] != 1:
            count+= 1
        if self.board[y][x-1] != 1:
            count+= 1
        if self.board[y+1][x] != 1:
            count+= 1
        if self.board[y-1][x] != 1:
            count+= 1

        return count
