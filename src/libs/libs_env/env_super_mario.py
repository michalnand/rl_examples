import sys
sys.path.append("..") # Adds higher directory to python modules path.

from nes_py.wrappers import JoypadSpace
from gym_super_mario_bros.actions import SIMPLE_MOVEMENT
import gym_super_mario_bros

import random
import time
import numpy
import math

import libs.libs_env.env
import libs.libs_gl_gui.gl_gui as libs_gl_gui


class EnvSuperMario(libs.libs_env.env.Env):

    def __init__(self):

        libs.libs_env.env.Env.__init__(self)

        self.game = gym_super_mario_bros.make('SuperMarioBros-v0')
        self.game = JoypadSpace(self.game, SIMPLE_MOVEMENT)
        #self.game.set_sound_enabled(True)
        self.game.reset()

        state, reward, done, info = self.game.step(0)
        self.raw_state_shape = numpy.shape(state)

        self.reset()

        self.actions_count  = 7

        self.time_steps     = 4
        self.colors_count   = 3

        self.width_scale    = 4
        self.height_scale   = 3
        self.width_scaled  = self.raw_state_shape[1]//self.width_scale
        self.height_scaled = self.raw_state_shape[0]//self.height_scale

        self.width  = self.width_scaled
        self.height = self.height_scaled
        self.depth  = self.colors_count*self.time_steps
        self.observation_init()


        print("raw state shape ", self.raw_state_shape[1], self.raw_state_shape[0])
        print("scaling factor ", self.width_scale, self.height_scale)
        print("rescaled state ", self.width_scaled, self.height_scaled)
        print("state ", self.width, self.height)

        self.set_no_terminal_state()


        self.game_idx = 0

        size = self.colors_count*self.height*self.width

        self.state_vec_frame_0 = numpy.zeros(size)
        self.state_vec_frame_1 = numpy.zeros(size)
        self.state_vec_frame_2 = numpy.zeros(size)
        self.state_vec_frame_3 = numpy.zeros(size)



        self.size_ratio = self.width/self.height
        self.gui = libs_gl_gui.GLVisualisation()
        self.window_name = "super mario"

    def reset(self):
        self.game.reset()
        state, reward, done, info = self.game.step(0)


    def do_action(self, action):
        state, reward, done, info = self.game.step(action)

        self.set_no_terminal_state()
        if done:
            self.set_terminal_state()
            self.reset()
            self.game_idx+= 1

        self.reward = float(reward)

        self.update_state(state)
        self.next_move()



    def _print(self):
        print("done game ", self.game_idx, self.get_move(), " score ", self.get_score())

    def get_games_count(self):
        return self.game_idx

    def get_iterations(self):
        return self.get_move()



    def update_state(self, raw_state):
        raw_state_downsampled = raw_state[::self.height_scale, ::self.width_scale, ::]/256.0
        raw_state_downsampled = numpy.swapaxes(raw_state_downsampled,1,0)
        raw_state_downsampled = numpy.swapaxes(raw_state_downsampled,2,0)

        self.state_vec_frame_3 = self.state_vec_frame_2.copy()
        self.state_vec_frame_2 = self.state_vec_frame_1.copy()
        self.state_vec_frame_1 = self.state_vec_frame_0.copy()
        self.state_vec_frame_0 = raw_state_downsampled.reshape(self.colors_count*self.height*self.width)

        #self.observation = numpy.concatenate((self.state_vec_frame_0, self.state_vec_frame_1, self.state_vec_frame_2, self.state_vec_frame_3))

    def render(self):
        self.game.render()

    def render_state(self, frame = 0):
        self.gui.init(self.window_name, 768, 768)
        self.gui.start()

        self.gui.push()
        self.gui.translate(0.0, 0.0, 0.45)


        self.gui.push()
        self.gui.set_color(0.1, 0.1, 0.1)
        self.gui.translate(0.0, 0.0, 0.0)
        self.gui.paint_square(4.0)
        self.gui.pop()

        element_size = 2.0/self.width

        for y in range(0, self.height):
            for x in range(0, self.width):
                self.gui.push()

                r = self.__get_item_from_observation(x, y, 0, frame)
                g = self.__get_item_from_observation(x, y, 1, frame)
                b = self.__get_item_from_observation(x, y, 2, frame)

                self.gui.set_color(r, g, b)
                self.gui.translate(self.__x_to_gui_x(x), self.__y_to_gui_y(y), 0.0)
                self.gui.paint_square(element_size)
                self.gui.pop()


        self.gui.pop()

        self.gui.finish()

    def __x_to_gui_x(self, x):
        return self.size_ratio*(x*1.0/self.width - 0.5)*2.0

    def __y_to_gui_y(self, y):
        return -(y*1.0/self.height - 0.5)*2.0

    def __get_item_from_observation(self, x, y, ch, frame):
        idx = ((frame*self.colors_count + ch)*self.height + y)*self.width + x

        return self.observation[idx]
