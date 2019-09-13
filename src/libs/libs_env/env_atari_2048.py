import libs.libs_env.env_atari_interface as env_atari_interface
import libs.libs_gl_gui.gl_gui as libs_gl_gui


import numpy
import time
import random


class EnvAtari2048(env_atari_interface.EnvAtariInterface):

    def __init__(self, size = 32):
        env_atari_interface.EnvAtariInterface.__init__(self, size)

        #5 actions for movements
        self.actions_count  = 5

        self.game_height = self.height
        self.game_width  = self.width

        #init game
        self.reset()

        self.window_name = "2048"



    def reset(self):
        self.__respawn()
        self.__update_state()
        self.game_cnt = 0
        self.board_prev = self.board.copy()


    def _print(self):
        #print("move=", self.get_move(), "  score=", self.get_score(), "  normalised score=", self.get_normalised_score())
        print("done game ", self.game_idx, " score ", self.get_score(), " max tile ", self.__board_max())

    def do_action(self, action):

        points = 0
        if action == 0:
            self.board = self.board_prev.copy()
        elif action == 1:
            points = self.__swipe_move(1)
        elif action == 2:
            points = self.__swipe_move(2)
        elif action == 3:
            points = self.__swipe_move(3)
        elif action == 4:
            points = self.__swipe_move(4)

        self.reward = points/8.0
        self.set_no_terminal_state()

        if self.__board_max() == 4096:
            self.reward = 10.0
            self.set_terminal_state()
            self.__respawn()
        elif self.__count_empty() == 0:
            self.reward = -10.0
            self.set_terminal_state()
            self.__respawn()
            self.game_cnt+= 1
            if self.game_cnt >= 20:
                self.next_game()
                self.game_cnt = 0
        else:
            self.board_prev = self.board.copy()
            self.__spawn_element(2)

        self.__update_state()
        self.update_state()
        self.next_move()


    def __swipe_move(self, move):
        points = 0
        for i in range(0, self.board_size):
            if move == 1:
                points+= self.__swipe_row_right(i)
            elif move == 2:
                points+= self.__swipe_row_left(i)
            elif move == 3:
                points+= self.__swipe_column_right(i)
            elif move == 4:
                points+= self.__swipe_column_left(i)

        return points

    def __swipe_row_right(self, row):
        change = True

        merged = 0
        while change:
            change = False
            for x in range(0, self.board_size-1):
                x_ = self.board_size- 1 - x

                if self.board[row][x_] == 0 and self.board[row][x_-1] != 0:
                    self.board[row][x_] = self.board[row][x_-1]
                    self.board[row][x_-1] = 0
                    change = True
                elif self.board[row][x_-1] != 0 and self.board[row][x_] == self.board[row][x_-1]:
                    self.board[row][x_]*= 2
                    self.board[row][x_-1] = 0
                    change = True
                    merged+= 1

        return merged

    def __swipe_row_left(self, row):
        change = True
        merged = 0
        while change:
            change = False
            for x in range(0, self.board_size-1):
                x_ = x

                if self.board[row][x_] == 0 and self.board[row][x_ + 1] != 0:
                    self.board[row][x_] = self.board[row][x_ + 1]
                    self.board[row][x_ + 1] = 0
                    change = True
                elif self.board[row][x_ + 1] != 0 and self.board[row][x_] == self.board[row][x_ + 1]:
                    self.board[row][x_]*= 2
                    self.board[row][x_ + 1] = 0
                    change = True
                    merged+= 1

        return merged


    def __swipe_column_right(self, column):
        change = True

        merged = 0
        while change:
            change = False
            for y in range(0, self.board_size-1):
                y_ = self.board_size- 1 - y

                if self.board[y_][column] == 0 and self.board[y_-1][column] != 0:
                    self.board[y_][column] = self.board[y_-1][column]
                    self.board[y_-1][column] = 0
                    change = True
                elif self.board[y_-1][column] != 0 and self.board[y_][column] == self.board[y_-1][column]:
                    self.board[y_][column]*= 2
                    self.board[y_-1][column] = 0
                    change = True
                    merged+= 1

        return merged

    def __swipe_column_left(self, column):
        change = True
        merged = 0
        while change:
            change = False
            for y in range(0, self.board_size-1):
                y_ = y

                if self.board[y_][column] == 0 and self.board[y_ + 1][column] != 0:
                    self.board[y_][column] = self.board[y_ + 1][column]
                    self.board[y_ + 1][column] = 0
                    change = True
                elif self.board[y_ + 1][column] != 0 and self.board[y_][column] == self.board[y_ + 1][column]:
                    self.board[y_][column]*= 2
                    self.board[y_ + 1][column] = 0
                    change = True
                    merged+= 1

        return merged


    def __respawn(self):
        self.board_size = 4

        self.board = [[0 for x in range(self.board_size)] for y in range(self.board_size)]

        self.__spawn_element(2)
        self.__spawn_element(2)

        self.board_prev = self.board.copy()



    def __update_state(self):
        self.clear_game_screen()

        for y in range(0, self.height):
            for x in range(0, self.width):
                color = [0.5, 0.5, 0.5]
                self.set_game_screen_element(x, y, color)

        offset_x = self.width//2 - self.width//4
        offset_y = self.height//2 - self.height//4

        element_size = 2
        pattern_size = 3

        self.board_max_value = self.__board_max()

        for y in range(0, self.board_size):
            for x in range(0, self.board_size):
                if self.board[y][x] != 0:
                    pattern = self.__create_element_mat(self.board[y][x]).copy()
                    for py in range(0, pattern_size):
                        for px in range(0, pattern_size):
                            color = pattern[py][px]
                            for ey in range(0, element_size):
                                for ex in range(0, element_size):
                                    x_ = offset_x + ex + px*element_size + x*element_size*pattern_size
                                    y_ = offset_y + ey + py*element_size + y*element_size*pattern_size
                                    self.set_game_screen_element(x_, y_, color)


    def __spawn_element(self, value, x = -1, y = -1):

        if x == -1 and y == -1:
            if self.__count_empty() == 0:
                return -1


            max_x = len(self.board[0]) - 1
            max_y = len(self.board) - 1

            x = random.randint(0, max_x)
            y = random.randint(0, max_y)

            while self.board[y][x] != 0:
                x = random.randint(0, max_x)
                y = random.randint(0, max_y)

        self.board[y][x] = value

        return 0

    def __count_empty(self):
        empty_field = 0
        for y in range(0, self.board_size):
            for x in range(0, self.board_size):
                if self.board[y][x] == 0:
                    empty_field+= 1
        return empty_field


    def __board_max(self):
        result  = 0
        for y in range(0, self.board_size):
            for x in range(0, self.board_size):
                if self.board[y][x] > result:
                    result = self.board[y][x]
        return result

    def __create_element_mat(self, value):

        alpha = numpy.log2(value)/numpy.log2(self.board_max_value)

        result = [[[alpha, 0.0, 1.0 - alpha] for x in range(3)] for y in range(3)]

        return result


    '''
    def __create_element_mat(self, value):

        if value == 2:
            type = 1
            color_id = 5

        elif value == 4:
            type = 2
            color_id = 5

        elif value == 8:
            type = 3
            color_id = 5

        elif value == 16:
            type = 4
            color_id = 5



        elif value == 32:
            type = 1
            color_id = 6

        elif value == 64:
            type = 2
            color_id = 6

        elif value == 128:
            type = 3
            color_id = 6

        elif value == 256:
            type = 4
            color_id = 6



        elif value == 512:
            type = 1
            color_id = 1

        elif value == 1024:
            type = 2
            color_id = 1

        elif value == 2048:
            type = 3
            color_id = 1

        elif value == 4096:
            type = 4
            color_id = 1

        result = [[[1.0, 1.0, 1.0] for x in range(3)] for y in range(3)]


        if type == 1:
            result[1][1] = self.item_to_color(color_id)
        elif type == 2:
            result[1][0] = self.item_to_color(color_id)
            result[1][2] = self.item_to_color(color_id)
        elif type == 3:
            result[0][0] = self.item_to_color(color_id)
            result[0][2] = self.item_to_color(color_id)
            result[2][0] = self.item_to_color(color_id)
            result[2][2] = self.item_to_color(color_id)
        elif type == 4:
            result[0][0] = self.item_to_color(color_id)
            result[0][1] = self.item_to_color(color_id)
            result[0][2] = self.item_to_color(color_id)
            result[1][0] = self.item_to_color(color_id)
            result[1][1] = self.item_to_color(color_id)
            result[1][2] = self.item_to_color(color_id)
            result[2][0] = self.item_to_color(color_id)
            result[2][1] = self.item_to_color(color_id)
            result[2][2] = self.item_to_color(color_id)


        return result
    '''
