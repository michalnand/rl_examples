import sys
sys.path.append("..") # Adds higher directory to python modules path.

from vizdoom import *
import random
import time
import numpy
import math

import libs.libs_env.env
import libs.libs_gl_gui.gl_gui as libs_gl_gui

import frame_buffer

class EnvDoom(libs.libs_env.env.Env):

    def __init__(self, mode = "basic"):


        libs.libs_env.env.Env.__init__(self)

        self.game = DoomGame()

        self.mode = mode

        self.positive_reward_factor = 1.0
        self.negative_reward_factor = 1.0

        if self.mode == "basic":
            self.positive_reward_factor = 0.1
            self.negative_reward_factor = 0.1

            self.game.load_config("scenarios/basic.cfg")

            left    = [1, 0, 0]
            right   = [0, 1, 0]
            attack  = [0, 0, 1]

            self.actions = [left, right, attack]

        if self.mode == "health_gathering":
            self.positive_reward_factor = 0.02
            self.negative_reward_factor = 0.1

            self.game.load_config("scenarios/health_gathering.cfg")

            turn_left       = [1, 0, 0]
            turn_right      = [0, 1, 0]
            move_forward    = [0, 0, 1]
            self.actions = [turn_left, turn_right, move_forward]

        if self.mode == "defend_the_center":
            self.positive_reward_factor = 5.0
            self.negative_reward_factor = 10.0

            self.game.load_config("scenarios/defend_the_center.cfg")

            turn_left   = [1, 0, 0]
            turn_right  = [0, 1, 0]
            attack      = [0, 0, 1]
            self.actions = [turn_left, turn_right, attack]


        if self.mode == "defend_the_line":
            self.positive_reward_factor = 1.0
            self.negative_reward_factor = 10.0

            self.game.load_config("scenarios/defend_the_line.cfg")

            left    = [1, 0, 0]
            right   = [0, 1, 0]
            attack  = [0, 0, 1]

            self.actions = [left, right, attack]


        if self.mode == "deadly_corridor":
            self.positive_reward_factor = 1.0
            self.negative_reward_factor = 0.5

            self.game.load_config("scenarios/deadly_corridor.cfg")

            move_left       = [1, 0, 0, 0, 0, 0, 0]
            move_right      = [0, 1, 0, 0, 0, 0, 0]
            attack          = [0, 0, 1, 0, 0, 0, 0]
            move_forward    = [0, 0, 0, 1, 0, 0, 0]
            move_backward   = [0, 0, 0, 0, 1, 0, 0]
            turn_left       = [0, 0, 0, 0, 0, 1, 0]
            turn_right      = [0, 0, 0, 0, 0, 0, 1]

            self.actions = [move_left, move_right, attack, move_forward, move_backward, turn_left, turn_right]


        if self.mode == "multi":
            self.positive_reward_factor = 0.1
            self.negative_reward_factor = 0.1

            self.game.load_config("scenarios/multi.cfg")

            turn_left       = [1, 0, 0, 0, 0, 0, 0, 0, 0]
            turn_right      = [0, 1, 0, 0, 0, 0, 0, 0, 0]
            attack          = [0, 0, 1, 0, 0, 0, 0, 0, 0]

            move_right      = [0, 0, 0, 1, 0, 0, 0, 0, 0]
            move_left       = [0, 0, 0, 0, 1, 0, 0, 0, 0]
            move_forward    = [0, 0, 0, 0, 0, 1, 0, 0, 0]
            move_backward   = [0, 0, 0, 0, 0, 0, 1, 0, 0]

            turn_left_right_delta = [0, 0, 0, 0, 0, 0, 0, 1, 0]
            move_left_right_delta = [0, 0, 0, 0, 0, 0, 0, 0, 1]

            self.actions = [ turn_left, turn_right, attack,
                             move_right, move_left, move_forward, move_backward,
                             turn_left_right_delta, move_left_right_delta]



        if self.mode == "deathmatch":
            self.positive_reward_factor = 1.0
            self.negative_reward_factor = 1.0

            self.game.load_config("scenarios/deathmatch.cfg")
            attack          = [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
            strafe          = [0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
            move_right      = [0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
            move_left       = [0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
            move_forward    = [0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
            move_backward   = [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
            turn_left       = [0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
            turn_right      = [0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]

            select_weapon_1 =    [0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
            select_weapon_2 =    [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0]
            select_weapon_3 =    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0]
            select_weapon_4 =    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0]
            select_weapon_5 =    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0]
            select_weapon_6 =    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0]
            select_next_weapon = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0]
            select_prev_weapon = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0]

            look_up_down_delta = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0]
            turn_left_right_delta=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0]
            move_left_right_delta=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1]

            self.actions = [    attack, strafe, move_right, move_left, move_forward, move_backward, turn_left, turn_right,
                                select_weapon_1, select_weapon_2, select_weapon_3, select_weapon_4, select_weapon_5, select_weapon_6,
                                select_next_weapon, select_prev_weapon, look_up_down_delta, turn_left_right_delta, move_left_right_delta]


        #self.game.add_available_game_variable(GameVariable.KILLCOUNT)
        #self.game.add_available_game_variable(GameVariable.DEATHCOUNT)

        #self.game.set_sound_enabled(True)
        self.game.init()
        self.game.new_episode()

        self.reset()

        self.raw_state_shape = self.game.get_state().screen_buffer.shape



        buffer_size = 64
        self.time_steps     = 8
        self.colors_count   = 3

        self.width  = 80
        self.height = 80
        self.depth  = self.colors_count*self.time_steps
        self.time   = 1

        self.width_scale    = 5
        self.height_scale   = 5

        self.width_scaled = self.raw_state_shape[2]//self.width_scale
        self.height_scaled = self.raw_state_shape[1]//self.height_scale

        self.width_crop = self.raw_state_shape[2]//self.width_scale     - self.width
        self.height_crop = self.raw_state_shape[1]//self.height_scale   - self.height


        print("raw state shape ", self.raw_state_shape[2], self.raw_state_shape[1])
        print("scaling factor ", self.width_scale, self.height_scale)
        print("rescaled state ", self.width_scaled, self.height_scaled)
        print("crop ", self.width_crop, self.height_crop)


        self.actions_count  = len(self.actions)
        self.observation_init()

        self.game_idx = 0

        self.frame_buffer = frame_buffer.FrameBuffer(self.width, self.height, self.colors_count, buffer_size)

        self.size_ratio = self.width/self.height
        self.gui = libs_gl_gui.GLVisualisation()
        self.window_name = "DOOM - " + self.mode

    def reset(self):
        self.kill_count  = 0
        self.death_count = 0

        self.game_kill_count  = 0
        self.game_death_count = 0

        self.game_kd_ratio = 0
        self.kd_ratio = 0

    def do_action(self, action):
        reward = self.game.make_action(self.actions[action])

        if reward > 0.0:
            self.reward = reward*self.positive_reward_factor
        else:
            self.reward = reward*self.negative_reward_factor


        if self.game.is_episode_finished():
            self.set_terminal_state()
            self.game.new_episode()
            self.game_idx+= 1

            self.kill_count+= self.game_kill_count
            self.death_count = self.game_death_count

            if self.kill_count > 20 and self.game_death_count > 10:
                self.game_kd_ratio = self.game_kill_count/(self.game_death_count + 0.00001)
                self.kd_ratio = self.kill_count/(self.death_count + 0.00001)

        else:
            self.set_no_terminal_state()

        state = self.game.get_state()
        screen_buffer = state.screen_buffer
        misc = state.game_variables

        self.game_kill_count  = misc[0]
        self.game_death_count = misc[1]

        self.update_state(screen_buffer)

        self.next_move()

    def _print(self):
        #print("move=", self.get_move(), "  score=", self.get_score(), "  normalised score=", self.get_normalised_score())
        print("done game ", self.game_idx, self.get_move(), " score ", self.get_score(),self.game_kill_count, self.game_death_count, self.kill_count, self.death_count, self.game_kd_ratio)

    def get_games_count(self):
        return self.game_idx

    def get_iterations(self):
        return self.get_move()

    def get_kill_count(self):
        return self.kill_count

    def get_death_count(self):
        return self.death_count

    def get_game_kd_ratio(self):
        return self.game_kd_ratio

    def get_kd_ratio(self):
        return self.kd_ratio

    def update_state(self, raw_state):
        raw_state_downsampled = raw_state[::, ::self.height_scale, ::self.width_scale]/256.0

        start_x = self.width_crop//2
        end_x   = self.width_scaled - self.width_crop//2
        start_y = self.height_crop//2
        end_y   = self.height_scaled - self.height_crop//2
        raw_state_cropped = raw_state_downsampled[::, start_y:end_y, start_x:end_x]

        self.frame_buffer.add_item(raw_state_cropped.reshape(self.colors_count*self.height*self.width))
        self.observation = numpy.concatenate((self.frame_buffer.get(0), self.frame_buffer.get(1), self.frame_buffer.get(2), self.frame_buffer.get(3), self.frame_buffer.get(7), self.frame_buffer.get(15), self.frame_buffer.get(31), self.frame_buffer.get(63)))


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
