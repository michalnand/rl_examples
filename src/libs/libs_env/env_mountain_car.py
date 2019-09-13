import libs.libs_env.env as env
import libs.libs_gl_gui.gl_gui as gl_gui

import numpy
import time
import random



class EnvMountainCar(env.Env):

    def __init__(self):

        #init parent class -> environment interface
        env.Env.__init__(self)

        #dimensions 1x1x2
        self.width  = 1
        self.height = 1
        self.depth  = 2

        #init state, as 1D vector (tensor with size depth*height*width)
        self.observation_init()

        #4 actions for movements
        self.actions_count  = 3

        self.velocity_min = -0.07
        self.velocity_max =  0.07
        self.position_min = -1.2
        self.position_max =  0.6001

        self.move_old = 0
        self.move_to_top = 0

        self.reset()

        self.gui = gl_gui.GLVisualisation()


    def reset(self):

        rnd = (random.random() - 0.5)*2.0
        self.position = -0.5 #+ 0.01*rnd
        self.velocity = 0.0 #0.01*rnd

        self.__update_observation()

    def _print(self):
        print("move=", self.get_move(), "  score=", self.get_score(), "  normalised score=", self.get_normalised_score(), " moves to top = ", self.move_to_top)
        self.render()

    def render(self):
        self.gui.init("mountain car")
        self.gui.start()

        self.gui.push()
        self.gui.translate(0.0, 0.0, -0.01)
        self.gui.set_color(0.0, 0.0, 0.7)
        self.gui.paint_square(2.0)
        self.gui.pop()

        element_size = 0.3

        elements = 1000

        y_scale = 0.5

        for i in range(0, elements):

            x = self.__map(i, 0.0, elements, self.position_min, self.position_max)
            y = numpy.sin(3.0*x)

            xp = self.__map(i, 0.0, elements, -1.0, 1.0)

            self.gui.push()
            self.gui.set_color(0.0, 0.7, 0.0)
            self.gui.paint_line(xp, -1.0, 0.0, xp, y_scale*y, 0.0)
            self.gui.pop()

        x = self.__map(self.position, self.position_min, self.position_max,  -0.5, 1.0) - element_size
        y = numpy.sin(3.0*self.position)

        self.gui.push()
        self.gui.translate(x, y_scale*y, 0.0)
        self.gui.set_color(1.0, 1.0, 1.0)
        self.gui.paint_textured_rectangle(element_size, element_size*0.5, 6)
        self.gui.pop()

        self.gui.finish()
        time.sleep(0.02)

    def do_action(self, action):

        if action == 0:
            acc = -1.0
        elif action == 1:
            acc = 1.0
        else:
            acc = 0.0

        velocity = self.velocity + acc*0.001 + numpy.cos(3.0*self.position)*(-0.0025) - 0.004*self.velocity
        position = self.position + self.velocity

        if position < self.position_min:
            position = self.position_min
            velocity = 0

        if position > self.position_max:
            position = self.position_max
            velocity = 0

        self.position = position
        self.velocity = velocity

        if self.position >= 0.6:
            self.reward = 1.0
            self.set_terminal_state()
            self.reset()

            self.move_to_top = self.move - self.move_old
            self.move_old = self.move
        else:
            self.set_no_terminal_state()
            self.reward = -0.001

        self.__update_observation()

        self.next_move()

    def get_move_to_top(self):
        return self.move_to_top

    def x_to_gui_x(self, x):
        return (x*1.0/self.width - 0.5)*2.0

    def y_to_gui_y(self, y):
        return (y*1.0/self.height - 0.5)*2.0


    def __saturate(self, value, min, max):
        if value > max:
            value = max

        if value < min:
            value = min

        return value

    def __map(self, value, source_min, source_max, dest_min, dest_max):
        k = (dest_max - dest_min)/(source_max - source_min)
        q = dest_min - k*source_min

        return k*value + q

    def __update_observation(self):
        self.observation[0] = self.__map(self.position, self.position_min, self.position_max, 0.0, 1.0)
        self.observation[1] = self.__map(self.velocity, self.velocity_min, self.velocity_max, 0.0, 1.0)

        for i in range(0, len(self.observation)):
            self.observation[i]+= random.random()*0.01
