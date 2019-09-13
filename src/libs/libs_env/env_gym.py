import gym

import numpy
import random

import libs.libs_env.env as libs_env
import libs.libs_gl_gui.gl_gui as libs_gl_gui



class EnvGym(libs_env.Env):

    def __init__(self, name):
        libs_env.Env.__init__(self)

        self.gym_env = gym.make(name)

        self.gym_env.reset()

        self.width_scale  = 3
        self.height_scale = 2

        self.raw_width  = numpy.shape(self.gym_env.observation_space)[0]
        self.raw_height = numpy.shape(self.gym_env.observation_space)[1]
        self.raw_depth  = numpy.shape(self.gym_env.observation_space)[2]

        self.time_steps   = 4
        self.colors_count = 3

        self.width      = 96
        self.height     = 96
        self.depth      = self.colors_count*self.time_steps

        self.actions_count = self.gym_env.action_space.n

        size = self.colors_count*self.height*self.width

        self.state_vec_frame_0 = numpy.zeros(size)
        self.state_vec_frame_1 = numpy.zeros(size)
        self.state_vec_frame_2 = numpy.zeros(size)
        self.state_vec_frame_3 = numpy.zeros(size)

        self.observation_init()

        self.gui = libs_gl_gui.GLVisualisation()
        self.game_number = 0



    def render(self):
        self.gym_env.render()

        '''
        self.gui.start()
        self.gui.translate(0, 0, 0.45)

        for y in range(0, self.height):
            for x in range(0, self.width):

                dim_max = self.width
                if self.height > dim_max:
                    dim_max = self.height

                size = 1.5/dim_max
                y_ = (y*1.0/dim_max - 0.5)*2.0
                x_ = (x*1.0/dim_max - 0.5)*2.0


                layer_size = self.width*self.height
                idx = y*self.width + x

                r = self.state_vec_frame_0[idx + layer_size*0]
                g = self.state_vec_frame_0[idx + layer_size*1]
                b = self.state_vec_frame_0[idx + layer_size*2]


                self.gui.push()

                self.gui.translate(x_, -y_, 0.0)

                self.gui.set_color(r, g, b)
                self.gui.paint_square(size)

                self.gui.pop()

        self.gui.finish()
        '''

    def do_action(self, action):

        observation, reward, done, info = self.gym_env.step(action)

        self.reward = reward


        if done:
            self.set_terminal_state()
            self.gym_env.reset()
            self.game_number+= 1
        else:
            self.set_no_terminal_state()

        self.__update_state(observation)
        self.next_move()

    def get_game_number(self):
        return self.game_number


    def __update_state(self, observation):

        observation_downsampled = observation[::self.width_scale, ::self.height_scale, ::]/256.0

        channels    = len(observation_downsampled[0][0])
        height      = len(observation_downsampled[0])
        width       = len(observation_downsampled)

        self.state_vec_frame_3 = self.state_vec_frame_2.copy()
        self.state_vec_frame_2 = self.state_vec_frame_1.copy()
        self.state_vec_frame_1 = self.state_vec_frame_0.copy()

        #self.state_vec_frame_0 = numpy.ones(self.width*self.height*3)*0.3

        x_offset = (self.width - width)//2
        y_offset = (self.height - height)//2
        for ch in range(0, 3):
            for y in range(0, height):
                for x in range(0, width):
                    v = observation_downsampled[x][y][ch]

                    idx_out = (ch*self.height + x + x_offset)*self.width + y + y_offset
                    self.state_vec_frame_0[idx_out] = v
        self.observation = numpy.concatenate((self.state_vec_frame_0, self.state_vec_frame_1, self.state_vec_frame_2, self.state_vec_frame_3))
