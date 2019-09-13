import libs.libs_env.env as env
import libs.libs_gl_gui.gl_gui as gl_gui

import numpy
import time
import random


class EnvBirds(env.Env):

    def __init__(self):
        env.Env.__init__(self)

        self.width  = 1
        self.height = 1
        self.depth  = 2

        self.actions_count = 2

        self.observation_init()

        self.reset()

        self.gui = gl_gui.GLVisualisation()

    def reset(self):

        rnd = random.random()*0.5 + 0.25

        self.s      = rnd   #position
        self.v      = 0.0   #vertical velocity

        self.hole_x     = 1.0
        self.hole_y     = 0.5
        self.hole_height = 1.0/4.0

        self.bot_size = 0.1


    def do_action(self, action):

        if action == 0:
            acc = 0.01
        else:
            acc = -0.01

        k = 0.6
        self.v = self.v*k + acc*(1.0 - k);
        self.s+= self.v

        '''
        if action == 0:
            f = 0.02 - 0.01
        else:
            f = 0.0  - 0.01

        self.v+= f
        self.s+= self.v*0.1
        '''

        if self.v > 1.0:
            self.v = 1.0

        if self.v < -1.0:
            self.v = -1.0

        if self.s > 1.0:
            self.s = 1.0

        if self.s < 0.0:
            self.s = 0.0


        self.reward = 0.0


        if self.hole_x > 0.0:
            self.hole_x-= 0.025
        else:
            center_distance = ((self.s - self.hole_y)**2.0)**0.5

            if center_distance < self.hole_height - self.bot_size:
                self.reward = 1.0
            else:
                self.reward = -1.0
                self.set_terminal_state()

                self.reset()

            self.new_hole()

        self.observation[0] = self.s
        self.observation[1] = self.hole_y


        self.next_move()

    def new_hole(self):
        self.hole_x = 1.0
        self.hole_y = random.random()*0.5 + 0.25


    def render_hole(self):


        '''
        x = self.x_to_gui_x(self.hole_x)
        y = self.y_to_gui_y(self.s)

        self.gui.push()
        self.gui.set_color(0.0, 1.0, 1.0)
        self.gui.translate(x, -y/2.0, 0.0)
        self.gui.paint_rectangle(0.15, (y + 1.0)/2.0)
        self.gui.pop()
        '''

        '''
        y_top    = self.y_to_gui_y(self.hole_y - self.hole_height/2.0)
        height_top = 2.0*(1.0 - (self.hole_y + self.hole_height*0.5))

        y_bottom = self.y_to_gui_y(self.hole_y + self.hole_height/2.0)
        height_bottom = 2.0*(self.hole_y - self.hole_height*0.5)

        self.gui.push()
        self.gui.set_color(0.0, 1.0, 0.0)
        self.gui.translate(x, y_top, 0.0)
        self.gui.paint_rectangle(0.15, height_top)
        self.gui.pop()

        self.gui.push()
        self.gui.set_color(0.0, 1.0, 1.0)
        self.gui.translate(x, y_bottom, 0.0)
        self.gui.paint_rectangle(0.15, height_bottom)
        self.gui.pop()
        '''


        y_top = (1.0 + (self.hole_y + self.hole_height*0.5))/2.0
        height_top = 2.0*(1.0 - (self.hole_y + self.hole_height*0.5))

        y_bottom = (self.hole_y - self.hole_height*0.5)/2.0
        height_bottom = 2.0*(self.hole_y - self.hole_height*0.5)

        y_top    = self.y_to_gui_y(y_top)
        y_bottom = self.y_to_gui_y(y_bottom)

        if height_top > 0.0 and height_top < 1.0:
            self.gui.push()
            self.gui.set_color(0.0, 1.0, 0.0)
            self.gui.translate(x, y_top, 0.0)
            self.gui.paint_rectangle(0.15, height_top)
            self.gui.pop()

        if height_bottom >= 0.0 and height_bottom <= 1.0:
            self.gui.push()
            self.gui.set_color(0.0, 1.0, 0.0)
            self.gui.translate(x, y_bottom, 0.0)
            self.gui.paint_rectangle(0.15, height_bottom)
            self.gui.pop()



    def render(self):


        self.gui.init("flappy birds")

        self.gui.start()

        self.gui.push()
        self.gui.translate(0.0, 0.0, -0.01)
        self.gui.set_color(0.0, 0.0, 0.8)
        self.gui.paint_square(3.0)
        self.gui.pop()


        player_x = self.x_to_gui_x(0.0)
        player_y = self.y_to_gui_y(self.s)


        self.gui.push()
        self.gui.translate(player_x, player_y, 0.0)
        self.gui.set_color(1.0, 1.0, 0.0)
        self.gui.paint_square(self.bot_size)
        self.gui.pop()

        self.render_hole()

        self.gui.push()
        self.gui.set_color(0.0, 0.0, 0.0)
        self.gui.translate(0.0, 1.0, 0.0)
        self.gui.paint_rectangle(2.0, 0.02)
        self.gui.pop()

        self.gui.push()
        self.gui.set_color(0.0, 0.0, 0.0)
        self.gui.translate(0.0, -1.0, 0.0)
        self.gui.paint_rectangle(2.0, 0.02)
        self.gui.pop()


        self.gui.set_color(1.0, 1.0, 1.0)
        count = "SCORE " + str(round(self.get_score(), 0))
        self.gui._print(-0.3, 0.95, 0.1, count)


        self.gui.finish()

        time.sleep(0.02)


    def x_to_gui_x(self, x):
        return (x*1.0/1.0 - 0.5)*2.0

    def y_to_gui_y(self, y):
        return (y*1.0/1.0 - 0.5)*2.0
