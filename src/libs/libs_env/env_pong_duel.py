import libs.libs_env.env as env
import libs.libs_gl_gui.gl_gui as gl_gui

import numpy
import time
import random



class EnvPongDuel(env.Env):

    def __init__(self):

        #init parent class -> environment interface
        env.Env.__init__(self)

        #dimensions 1x10x10
        self.width  = 20
        self.height = 16
        self.depth  = 1

        #init state, as 1D vector (tensor with size depth*height*width)
        self.observation_init()

        #4 actions for movements
        self.actions_count  = 2

        self.player_size = 5

        self.reset()

        self.gui = gl_gui.GLVisualisation()


        self.size_ratio = self.width/self.height

        self.player_0_score = 0
        self.player_1_score = 0



    def reset(self):
        #initial players position
        self.player_0 = self.height/2
        self.player_1 = self.height/2

        #ball to center + some noise
        self.ball_x  = self.width/2  + random.randint(-1, 1)
        self.ball_y  = self.height/2 + random.randint(-1, 1)

        #random ball move

        if random.randint(0, 1) == 0:
            self.ball_dx = 1
        else:
            self.ball_dx = -1

        if random.randint(0, 1) == 0:
            self.ball_dy = 1
        else:
            self.ball_dy = -1

        self.active_player = 0

        self.__position_to_state()


    def _print(self):
        print("move=", self.get_move(), "  score=", self.get_score(), "  normalised score=", self.get_normalised_score())
        self.render()

    def render(self):
        score_ratio = round(self.player_0_score/(self.player_0_score + self.player_1_score + 0.00000000001), 3)
        print("score : ", self.player_0_score, self.player_1_score, score_ratio)

        self.gui.init("pong", 32*self.width, 32*self.height)

        if self.width > self.height:
            element_size = 2.0/self.width
        else:
            element_size = 2.0/self.height

        self.gui.start()

        for y in range(0, self.height):
            self.gui.push()
            self.gui.translate(self.x_to_gui_x(self.width/2), self.y_to_gui_y(y), 0.0)
            self.gui.set_color(1.0, 1.0, 1.0)
            self.gui.paint_textured_rectangle(element_size*0.5, element_size, 5)
            self.gui.pop()

        self.gui.push()
        self.gui.translate(self.x_to_gui_x(self.ball_x), self.y_to_gui_y(self.ball_y), 0.0)
        self.gui.set_color(1.0, 1.0, 1.0)
        self.gui.paint_textured_rectangle(element_size, element_size, 5)
        self.gui.pop()

        for y in range(0, self.player_size):
            self.gui.push()
            self.gui.translate(self.x_to_gui_x(self.width-1), self.y_to_gui_y(y + self.player_1 - self.player_size/2 + 1), 0.0)
            self.gui.set_color(1.0, 1.0, 1.0)
            self.gui.paint_textured_rectangle(element_size, element_size, 5)
            self.gui.pop()

        for y in range(0, self.player_size):
            self.gui.push()
            self.gui.translate(self.x_to_gui_x(0), self.y_to_gui_y(y + self.player_0 - self.player_size/2 + 1), 0.0)
            self.gui.set_color(1.0, 1.0, 1.0)
            self.gui.paint_textured_rectangle(element_size, element_size, 5)
            self.gui.pop()

        self.gui.set_color(1.0, 1.0, 1.0)
        count = "SCORE " + str(round(self.player_0_score, 0)) + ":" + str(round(self.player_1_score, 0))
        self.gui._print(-0.3, 0.95, 0.1, count)

        self.gui.finish()

        time.sleep(0.05)

    def do_action(self, action):
        self.reward = 0.0
        self.set_no_terminal_state()

        if self.active_player == 0:
            if action == 0:
                self.player_0+= 1
            else:
                self.player_0-= 1

            self.player_0 = self.__saturate(self.player_0, 0, self.height-1)
            self.active_player = 1
        else:
            if action == 0:
                self.player_1+= 1
            else:
                self.player_1-= 1
            self.player_1 = self.__saturate(self.player_1, 0, self.height-1)
            self.active_player = 0


        self.ball_x+= self.ball_dx
        self.ball_y+= self.ball_dy

        if self.ball_x <= 0:

            dif = numpy.absolute(self.player_0 - self.ball_y)
            if dif < self.player_size:
                self.ball_dx = 1
            else:
                self.reset()
                self.player_1_score+= 1


        if self.ball_x >=  self.width-1:
            dif = numpy.absolute(self.player_1 - self.ball_y)
            if dif < self.player_size:
                self.ball_dx = -1
            else:
                self.reset()
                self.player_0_score+= 1

        if self.ball_y <= 0:
            self.ball_dy = 1

        if self.ball_y >=  self.height-1:
            self.ball_dy = -1

        self.__position_to_state()
        self.next_move()

    def x_to_gui_x(self, x):
        return self.size_ratio*(x*1.0/self.width - 0.5)*2.0

    def y_to_gui_y(self, y):
        return (y*1.0/self.height - 0.5)*2.0

    def __position_to_state(self):
        if self.active_player == 0:
            ball_x = self.__saturate(int(self.ball_x), 0, self.width-1)
            ball_y = self.__saturate(int(self.ball_y), 0, self.height-1)

            player = self.__saturate(int(self.player_0), 0, self.height-1)

            self.observation.fill(0.0)
            self.observation[ball_y*self.get_width() + ball_x] = 1.0
            self.observation[player*self.get_width() + 0] = 1.0
        else:
            ball_x = (self.width-1) - self.__saturate(int(self.ball_x), 0, self.width-1)
            ball_y = self.__saturate(int(self.ball_y), 0, self.height-1)

            player = self.__saturate(int(self.player_1), 0, self.height-1)

            self.observation.fill(0.0)
            self.observation[ball_y*self.get_width() + ball_x] = 1.0
            self.observation[player*self.get_width() + 0] = 1.0

    def __saturate(self, value, min, max):
        if value > max:
            value = max

        if value < min:
            value = min

        return value
