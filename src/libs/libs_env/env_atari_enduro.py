import libs.libs_env.env_atari_interface as env_atari_interface
import libs.libs_gl_gui.gl_gui as libs_gl_gui

import numpy
import time
import random
import math

class EnvAtariEnduro(env_atari_interface.EnvAtariInterface):

    def __init__(self, size = 32):
        env_atari_interface.EnvAtariInterface.__init__(self, size)

        #3 actions for movements
        self.actions_count  = 3


        self.game_height = (int)(self.height)
        self.game_width  = (int)(self.width)

        #init game
        self.reset()

        self.window_name = "ENDURO"


    def reset(self):

        self.road_width = self.game_width//2
        self.road_state = 0
        self.road_position = self.game_width//2

        self.road_length = int(self.game_height*0.9)

        self.road = []
        for i in range(0, self.road_length):
            self.road.append(self.road_position)

        for i in range(0, self.road_length*self.road_length):
            self.__road_step()

        self.car_position = self.road[self.road_length-1]

        self.road_distance = 0



    def _print(self):
        print("move=", self.get_move(), "  score=", self.get_score(), " game=", self.get_games_count())

    def do_action(self, action):

        road_pos = self.road_length - 1

        '''
        if self.road[road_pos] > self.car_position:
            action = 1
        else:
            action = 2
        '''

        if action == 0:
            self.car_position = self.car_position
        elif action == 1:
            self.car_position+= 1
        elif action == 2:
            self.car_position-= 1

        self.car_position = self.__saturate(self.car_position, self.road_width//8, self.width-self.road_width//8)

        self.__road_step()


        self.set_no_terminal_state()

        self.reward = 0.0

        if abs(self.car_position - self.road[road_pos]) < self.road_width//2:
            self.reward = 0.025
            self.road_distance+= 1

            if self.road_distance > 500:
                self.road_distance = 0
                self.reward = 10.0
                self.set_terminal_state()
                self.next_game()
        else:
            self.road_distance = 0
            self.reward = -10.0
            self.set_terminal_state()
            self.reset()

        self.clear_game_screen()

        self.__put_ground()
        self.__put_horizont()
        self.__put_road()
        self.__put_car()

        self.update_state()
        self.next_move()



    def __put_road(self):
        color = self.item_to_color(10)

        y_offset = self.game_height - len(self.road)

        for y in range(0, len(self.road)):
            x = self.road[y]
            x0 = x - self.road_width//2
            x1 = x + self.road_width//2

            (xp0, yp0) = self.__to_perspective(x0, y + y_offset, x)
            (xp1, yp1) = self.__to_perspective(x1, y + y_offset, x)

            self.set_game_screen_element(xp0, yp0, color)
            self.set_game_screen_element(xp1, yp1, color)

    def __put_ground(self):
        color = self.item_to_color(16)

        for y in range(0, self.game_height):
            for x in range(0, self.game_width):
                self.set_game_screen_element(x, y, color)

    def __put_horizont(self):
        size = self.game_height - len(self.road)

        color = self.item_to_color(6)

        for y in range(0, size):
            for x in range(0, self.game_width):
                self.set_game_screen_element(x, y, color)

        color = self.item_to_color(17)

        for x in range(0, self.game_width):

            v = math.sin(3.141592654*(0.2 + 1.1*(self.car_position - x)/self.game_width))

            v_ = int(abs(v*size))

            for y in range(0, v_):
                self.set_game_screen_element(x, size - y, color)


    def __put_car(self):
        x = self.car_position
        y = self.game_height - 8
        color = self.item_to_color(1)

        self.set_game_screen_element(x - 1, y + 0, color)
        self.set_game_screen_element(x + 1, y + 0, color)


        self.set_game_screen_element(x, y + 0, color)
        self.set_game_screen_element(x, y + 1, color)
        self.set_game_screen_element(x, y + 2, color)
        self.set_game_screen_element(x, y + 3, color)
        self.set_game_screen_element(x, y + 4, color)
        self.set_game_screen_element(x, y + 5, color)

        self.set_game_screen_element(x - 1, y + 5, color)
        self.set_game_screen_element(x + 1, y + 5, color)



    def __saturate(self, value, min, max):
        if value > max:
            value = max

        if value < min:
            value = min

        return value

    def __road_step(self):

        for i in range(1, len(self.road)):
            idx = len(self.road) - i
            self.road[idx] = self.road[idx-1]

        if random.random() < 0.15:
            self.road_state = random.randint(0, 2)

        if self.road_state == 0:
            self.road_position+= 0
        elif self.road_state == 1:
            self.road_position+= 1
        elif self.road_state == 2:
            self.road_position-= 1

        self.road_position = self.__saturate(self.road_position, self.road_width//2, self.width-self.road_width//2)

        self.road[0] = self.road_position

    def __to_perspective(self, x, y, x_center):
        alpha    = (y**2)/(self.game_height**2)


        x_result = alpha*x + (1.0 - alpha)*x_center
        y_result = y

        return (int(x_result), int(y_result))
