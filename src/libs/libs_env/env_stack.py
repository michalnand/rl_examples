import libs.libs_env.env as libs_env
import libs.libs_gl_gui.gl_gui as libs_gl_gui


import numpy
import time
import random


class EnvStack(libs_env.Env):

    def __init__(self):

        libs_env.Env.__init__(self)

        self.size       = 16
        self.max_depth  = 16

        self.width   = self.size
        self.height  = self.size
        self.depth   = 2

        self.observation_init()

        self.actions_count = 2

        self.reset()

        self.angle = 0.0
        self.gui = libs_gl_gui.GLVisualisation()

    def do_action(self, action):

        self.reward = -0.05

        self.set_no_terminal_state()

        if self.actual_floor+1 >= self.max_depth:
            self.set_terminal_state()
            self.reset()

        if action == 1:
        #if random.random() < 0.1:
            overlap = self.__compute_floor_overlap(self.floors[self.actual_floor-1], self.floors[self.actual_floor])


            #if overlap > 0.8:
            if overlap > 0.9:
                self.reward = 1.0
            else:
                self.reward = -1.0

            #self.reward = overlap*2.0 - 1.0

            self.actual_floor+= 1
            self.x = random.randint(0, self.width - self.actual_width)



        self.__fill_floor(self.actual_floor, self.x, self.y, self.actual_width, self.actual_height)

        self.x+= self.dx
        self.y+= self.dy

        if self.x >= self.width - self.actual_width:
            self.dx = -1

        if self.x <= 0:
            self.dx = 1

        self.__update_state(self.floors[self.actual_floor], self.floors[self.actual_floor-1])
        self.next_move()

    def reset(self):
        self.actual_width   = self.width//2
        self.actual_height  = self.height//2
        self.actual_floor   = 1

        self.floors = numpy.zeros((self.max_depth, self.height, self.width))

        x_center = self.width//4
        y_center = self.height//4

        for y in range(0, self.actual_height):
            for x in range(0, self.actual_width):
                self.floors[0][y + y_center][x + x_center] = 1.0

        self.dx = 1
        self.dy = 0
        self.x  = x_center
        self.y  = y_center

        self.__fill_floor(self.actual_floor, self.x, self.y, self.actual_width, self.actual_height)



    def render(self):
        self.gui.start()

        element_size = 2.0/self.width

        self.gui.push()

        self.gui.translate(0.0, 0.0, -3.0)
        self.gui.rotate(-60.0, 0.0, 40.0 + self.angle)

        self.angle+= 1.0

        for z in range(0, self.max_depth):
            for y in range(0, self.height):
                for x in range(0, self.width):
                    if self.floors[z][y][x] > 0.0:

                        x_ = x*2.0/self.width       - 1.0
                        y_ = y*2.0/self.height      - 1.0
                        z_ = z*2.0/self.max_depth   - 1.0

                        phi   = 2.0*3.141592654*z*1.0/self.max_depth
                        phase = (2.0*3.141592654)/3.0

                        r = (numpy.sin(phi + phase*0) + 1.0)/2.0
                        g = (numpy.sin(phi + phase*1) + 1.0)/2.0
                        b = (numpy.sin(phi + phase*2) + 1.0)/2.0

                        self.gui.push()
                        self.gui.translate(x_, y_, z_)
                        self.gui.set_color(r, g, b)
                        self.gui.paint_cube(element_size)
                        self.gui.pop()

        self.gui.pop()
        self.gui.finish()

        time.sleep(0.05)

        #overlap = self.__compute_floor_overlap(self.floors[self.actual_floor-1], self.floors[self.actual_floor])
        #print("overlap = ", overlap)
        print("score = ", self.get_score())


    def __fill_floor(self, floor, x_center, y_center, actual_width, actual_height):

        self.floors[floor].fill(0.0)

        for y in range(0, actual_height):
            for x in range(0, actual_width):
                if y + y_center <  self.height:
                    if x + x_center <  self.width:
                        self.floors[floor][y + y_center][x + x_center] = 1.0



    def __update_state(self, floor_0, floor_1):

        self.observation.fill(0.0)

        idx = 0
        for y in range(0, self.height):
            for x in range(0, self.width):
                self.observation[idx] = floor_0[y][x]
                idx+= 1

        for y in range(0, self.height):
            for x in range(0, self.width):
                self.observation[idx] = floor_1[y][x]
                idx+= 1


    def __compute_floor_overlap(self, floor_0, floor_1):

        sum_max     = 0.0
        sum_overlap = 0.0

        for y in range(0, self.height):
            for x in range(0, self.width):

                if floor_0[y][x] > 0.0:
                    sum_max+= 1.0

                if floor_0[y][x] > 0.0 and floor_1[y][x] > 0.0:
                    sum_overlap+= 1.0

        overlap = sum_overlap/(sum_max + 0.0001)

        return overlap
