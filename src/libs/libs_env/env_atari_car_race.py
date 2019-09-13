import libs.libs_env.env_atari_interface as env_atari_interface
import libs.libs_gl_gui.gl_gui as libs_gl_gui


import numpy
import time
import random


class EnvAtariCarRace(env_atari_interface.EnvAtariInterface):

    def __init__(self):
        env_atari_interface.EnvAtariInterface.__init__(self)

        #3 actions for movements
        self.actions_count  = 5

        self.game_height = self.height
        self.game_width  = self.width

        #init game
        self.reset()
        self.window_name = "CAR RACE"



    def reset(self):
        self.clear_game_screen()

        self.road_center     = self.game_width//2
        self.road_state      = 1
        self.board           = numpy.zeros((self.game_height, self.game_width))
        self.player_position = self.game_width//2

        self.road_width = (int)(self.game_width*0.25)

        self.road = []
        for i in range(0, self.game_height):
            self.road.append(self.road_center)

            if random.random() < 0.05:
                self.road_state*= -1

            if self.road_center >=  self.game_width - self.road_width//2:
                self.road_state = -1
            if self.road_center <=  self.road_width//2:
                self.road_state = 1

            self.road_center+= self.road_state

        self.road_ptr = 0
        self.distance = 0


    def _print(self):
        #print("move=", self.get_move(), "  score=", self.get_score(), "  normalised score=", self.get_normalised_score())
        print("done game ", self.game_idx, " score ", self.get_score())

    def do_action(self, action):

        if action == 0:
            self.player_position+= 1
        elif action == 1:
            self.player_position-= 1
        if action == 2:
            self.player_position+= 2
        elif action == 3:
            self.player_position-= 2
        else:
            self.player_position+= 0

        self.player_position = self.__saturate(self.player_position, 0, self.game_width - 1)

        if random.random() < 0.1:
            self.road_state*= -1

        if self.road_center >=  self.game_width - self.road_width//2:
            self.road_state = -1
        if self.road_center <=  self.road_width//2:
            self.road_state = 1

        self.road_center+= self.road_state


        for j in range(0, len(self.road)-1):
            self.road[j] = self.road[j+1]
        self.road[len(self.road) - 1] = self.road_center

        player_y = 2

        self.clear_game_screen()

        for j in range(0, self.game_height):
            for i in range(0, self.game_width):
                self.board[j][i] = 0.0

        for j in range(0, self.game_height):
            for i in range(0, self.game_width):
                if (i > (self.road[j] - self.road_width/2)) and (i < self.road[j] + self.road_width/2):
                    self.board[j][i] = 1.0

        for j in range(0, self.game_height):
            for i in range(0, self.game_width):
                if self.board[j][i] > 0.0:
                    self.set_game_screen_element(i, j, 0, 0)
                    self.set_game_screen_element(i, j, 1, 0.8)
                    self.set_game_screen_element(i, j, 2, 0)


        self.board[self.player_position][player_y] = -1.0
        self.set_game_screen_element(self.player_position, player_y, 0, 0)
        self.set_game_screen_element(self.player_position, player_y, 1, 0)
        self.set_game_screen_element(self.player_position, player_y, 2, 1)


        self.reward = 0.1
        self.set_no_terminal_state()

        if self.player_position == self.road[player_y]:
            self.reward = 0.5

        if self.distance >= self.game_height*2:
            self.distance = 0
            self.reward = 1.0
            self.set_terminal_state()
            self.reset()
            self.next_game()
        else:
            self.distance+= 1

        if self.board[player_y][self.player_position] == 0.0:
            self.reward = -1.0
            self.distance = 0
            self.set_terminal_state()
            self.reset()


        self.update_state()
        self.next_move()


    def __saturate(self, value, min, max):
        if value > max:
            value = max

        if value < min:
            value = min

        return value
