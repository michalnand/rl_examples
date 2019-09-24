import libs.libs_env.env as env
import libs.libs_gl_gui.gl_gui as gl_gui

import libs.libs_rysy_python.rysy as rysy


import numpy
import time
import random


class Env2048(env.Env):

    def __init__(self):
        env.Env.__init__(self)

        self.width  = 4
        self.height = 4
        self.depth  = 1
        self.time   = 1

        self.actions_count = 4

        self.observation_init()
        self.reset()


        self.gui = gl_gui.GLVisualisation()

        self.max_value = 0.0
        self.max_score_log = rysy.Log("2048_max_score.log")


    def reset(self):
        self.__respawn()
        self.__update_state()
        self.game_idx   = 0
        self.board_prev = self.board.copy()


    def _print(self):
        #print("move=", self.get_move(), "  score=", self.get_score(), "  normalised score=", self.get_normalised_score())
        print("done game ", self.game_idx, " score ", self.get_score(), " max tile ", self.__get_board_max_value())

    def get_games_count(self):
        return self.game_idx

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

        self.reward = points/10.0
        self.set_no_terminal_state()

        if self.__get_board_max_value() > self.max_value:
            self.max_value = self.__get_board_max_value()

            max_score = str(self.get_move()) + " "
            max_score+= str(self.game_idx) + " "
            max_score+= str(self.max_value) + " "

            max_score+= "\n"

            self.max_score_log.put_string(max_score)


        if self.__get_board_max_value() == 2048:
            self.reward = 10.0
            self.game_idx+= 1
            self.set_terminal_state()
            self.__respawn()
        elif self.__count_empty() == 0:
            self.reward = -10.0
            self.game_idx+= 1
            self.set_terminal_state()
            self.__respawn()
        else:
            self.board_prev = self.board.copy()
            self.__spawn_element(2)

        self.__update_state()
        self.__update_state()
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
        board_max_value = self.__get_board_max_value()
        ptr = 0
        for y in range(0, self.height):
            for x in range(0, self.width):
                if self.board[y][x] > 0.0:
                    v = self.__get_normalised_value(self.board[y][x], board_max_value)
                else:
                    v = 0.0

                self.observation[ptr] = v
                ptr+= 1


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


    def __get_board_max_value(self):
        result  = 0
        for y in range(0, self.board_size):
            for x in range(0, self.board_size):
                if self.board[y][x] > result:
                    result = self.board[y][x]
        return result

    def __get_normalised_value(self, value, board_max_value):
        result = numpy.log2(value)/numpy.log2(board_max_value)

        return result


    def render(self):
        self.gui.init("2048")

        self.gui.start()

        board_max_value = self.__get_board_max_value()
        ptr = 0

        element_size = 1.0/self.width

        for y in range(0, self.height):
            for x in range(0, self.width):
                value = self.__get_normalised_value(self.board[y][x], board_max_value)

                x_ = element_size*2*(x*1.0/self.width - 0.5)*2.0
                y_ = element_size*2*(y*1.0/self.height - 0.5)*2.0

                self.gui.push()
                self.gui.translate(x_, y_, 0.0)
                self.gui.set_color(value, value, value)
                self.gui.paint_square(element_size)
                self.gui.pop()


        self.gui.set_color(1.0, 1.0, 1.0)
        count = "SCORE " + str(round(self.get_score(), 0))
        self.gui._print(-0.3, 0.95, 0.1, count)

        self.gui.finish()

        time.sleep(0.02)
