import libs.libs_env.env as env
import libs.libs_gl_gui.gl_gui as gl_gui

import numpy
import time
import random



class EnvSettlers(env.Env):

    def __init__(self):

        #init parent class -> environment interface
        env.Env.__init__(self)

        #dimensions 10x9x1
        self.width  = 9
        self.height = 10
        self.depth  = 1

        self.observation_init()

        self.actions_count  = 6

        self.moves_to_win   = 200
        self.game_moves     = 0.0

        self.reset()

        self.gui = gl_gui.GLVisualisation()


    def reset(self):

        #points to win game
        self.winning_points      = 10.0

        #resources initial state
        self.resources = { }
        self.resources["wood"]   =  0
        self.resources["brick"]  =  0
        self.resources["wool"]   =  0
        self.resources["crop"]   =  0
        self.resources["ore"]    =  0

        #initial items - village, road
        self.items = { }
        self.items["village"]   =   1
        self.items["city"]      =   0
        self.items["up city"]   =   0
        self.items["road"]      =   1
        self.items["knight"]    =   0

        #items costs
        self.costs = { }
        self.costs["pass"]      = [0, 0, 0, 0, 0]
        self.costs["village"]   = [1, 1, 1, 1, 0]
        self.costs["city"]      = [0, 0, 0, 2, 3]
        self.costs["up city"]   = [0, 0, 3, 0, 1]
        self.costs["road"]      = [1, 1, 0, 0, 0]
        self.costs["knight"]    = [0, 0, 1, 1, 1]

        #random items costs -> TODO test + costs visualisation
        random_costs = False
        if random_costs:
            costs_count = 4
            self.costs = { }
            self.costs["pass"]      = self.__random_costs(costs_count)
            self.costs["village"]   = self.__random_costs(costs_count)
            self.costs["city"]      = self.__random_costs(costs_count)
            self.costs["up city"]   = self.__random_costs(costs_count)
            self.costs["road"]      = self.__random_costs(costs_count)
            self.costs["knight"]    = self.__random_costs(costs_count)

        #points for buildable items
        #pass -> no item builded
        self.points = { }
        self.points["pass"]     = 0.0
        self.points["village"]  = 1.0
        self.points["city"]     = 2.0
        self.points["up city"]  = 3.0
        self.points["road"]     = 1.0
        self.points["knight"]   = 1.0

        for i in range(0, 3):
            self.resources[self.__get_card()]+= 1

        self.__update_observation()

    def _print(self):
        print("move=", self.get_move(), "  score=", self.get_score(), "  normalised score=", self.get_normalised_score(), "  moves to win =", self.moves_to_win)

    def render(self):
        self.gui.init("settlers")
        self.gui.start()

        element_size    = 0.4
        spacing         = 0.025

        self.gui.push()
        self.gui.translate(0.0, 0.0, -0.01)
        self.gui.set_color(1.0, 1.0, 1.0)
        self.gui.paint_textured_rectangle(3.0, 3.0, 12)
        self.gui.pop()

        self.gui.push()
        self.gui.translate(-0.8, 0.75, 0.0)


        self.gui.push()
        self.gui.translate(0.0*(element_size+spacing), 0.0, 0.0)
        self.gui.set_color(1.0, 1.0, 1.0)
        self.gui.paint_textured_rectangle(element_size, element_size, 7)
        count = str(self.resources["wood"])
        self.gui._print(-0.1, -0.3, 0.0, count);
        self.gui._print(-0.1, 0.3, 0.0, "WOOD");
        self.gui.pop()

        self.gui.push()
        self.gui.translate(1.0*(element_size+spacing), 0.0, 0.0)
        self.gui.set_color(1.0, 1.0, 1.0)
        self.gui.paint_textured_rectangle(element_size, element_size, 8)
        count = str(self.resources["brick"])
        self.gui._print(-0.1, -0.3, 0.0, count);
        self.gui._print(-0.1, 0.3, 0.0, "BRICK");
        self.gui.pop()

        self.gui.push()
        self.gui.translate(2.0*(element_size+spacing), 0.0, 0.0)
        self.gui.set_color(1.0, 1.0, 1.0)
        self.gui.paint_textured_rectangle(element_size, element_size, 9)
        count = str(self.resources["wool"])
        self.gui._print(-0.1, -0.3, 0.0, count);
        self.gui._print(-0.1, 0.3, 0.0, "WOOL");
        self.gui.pop()

        self.gui.push()
        self.gui.translate(3.0*(element_size+spacing), 0.0, 0.0)
        self.gui.set_color(1.0, 1.0, 1.0)
        self.gui.paint_textured_rectangle(element_size, element_size, 11)
        count = str(self.resources["crop"])
        self.gui._print(-0.1, -0.3, 0.0, count);
        self.gui._print(-0.1, 0.3, 0.0, "CROP");
        self.gui.pop()

        self.gui.push()
        self.gui.translate(4.0*(element_size+spacing), 0.0, 0.0)
        self.gui.set_color(1.0, 1.0, 1.0)
        self.gui.paint_textured_rectangle(element_size, element_size, 10)
        count = str(self.resources["ore"])
        self.gui._print(-0.1, -0.3, 0.0, count);
        self.gui._print(-0.1, 0.3, 0.0, "ORE");
        self.gui.pop()


        self.gui.pop()



        self.gui.push()
        self.gui.translate(-0.8, -0.75, 0.0)

        self.gui.push()
        self.gui.translate(0.0*(element_size+spacing), 0.0, 0.0)
        self.gui.set_color(1.0, 1.0, 1.0)
        self.gui.paint_textured_rectangle(element_size, element_size, 1)
        count = str(self.items["village"])
        self.gui._print(-0.1, -0.3, 0.0, count);
        self.gui._print(-0.1, 0.3, 0.0, "VILLAGE");
        self.gui.pop()

        self.gui.push()
        self.gui.translate(1.0*(element_size+spacing), 0.0, 0.0)
        self.gui.set_color(1.0, 1.0, 1.0)
        self.gui.paint_textured_rectangle(element_size, element_size, 1)
        count = str(self.items["city"])
        self.gui._print(-0.1, -0.3, 0.0, count);
        self.gui._print(-0.1, 0.3, 0.0, "CITY");
        self.gui.pop()

        self.gui.push()
        self.gui.translate(2.0*(element_size+spacing), 0.0, 0.0)
        self.gui.set_color(1.0, 1.0, 1.0)
        self.gui.paint_textured_rectangle(element_size, element_size, 1)
        count = str(self.items["up city"])
        self.gui._print(-0.1, -0.3, 0.0, count);
        self.gui._print(-0.1, 0.3, 0.0, "UP CITY");
        self.gui.pop()

        self.gui.push()
        self.gui.translate(3.0*(element_size+spacing), 0.0, 0.0)
        self.gui.set_color(1.0, 1.0, 1.0)
        self.gui.paint_textured_rectangle(element_size, element_size, 1)
        count = str(self.items["road"])
        self.gui._print(-0.1, -0.3, 0.0, count);
        self.gui._print(-0.1, 0.3, 0.0, "ROAD");
        self.gui.pop()

        self.gui.push()
        self.gui.translate(4.0*(element_size+spacing), 0.0, 0.0)
        self.gui.set_color(1.0, 1.0, 1.0)
        self.gui.paint_textured_rectangle(element_size, element_size, 1)
        count = str(self.items["knight"])
        self.gui._print(-0.1, -0.3, 0.0, count);
        self.gui._print(-0.1, 0.3, 0.0, "KNIGHT");
        self.gui.pop()

        self.gui.pop()


        score = self.__compute_score()

        count = "SCORE = " + str(score)
        self.gui._print(-1.0, -0.1, 0.0, count);
        count = "MOVES TO WIN = " + str(self.moves_to_win)
        self.gui._print(-1.0, -0.3, 0.0, count);


        self.gui.finish()
        time.sleep(0.1)

    def do_action(self, action):

        self.game_moves+= 1

        self.reward = -0.2 #-0.05

        if self.__is_legal_action(action):
            #execute action
            self.reward+= self.__execute_action(action)/self.winning_points

            #take next random card
            self.resources[self.__get_card()]+= 1
            '''
            for i in range(0, self.items["city"]):
                self.resources[self.__get_card()]+= 1

            for i in range(0, self.items["up city"]):
                self.resources[self.__get_card()]+= 2
            '''
        else:
            self.reward = -1.0

        self.__saturate_resources()
        self.__saturate_items()

        if self.__compute_score() >= self.winning_points:
            self.reward = 1.0

            k = 0.99
            self.moves_to_win = k*self.moves_to_win + (1.0 - k)*self.game_moves
            self.game_moves = 0.0

            self.set_terminal_state()
            self.reset()

        self.__update_observation()

        self.next_move()

    def get_moves_to_win(self):
        return self.moves_to_win

    def __update_observation(self):
        self.observation.fill(0.0)

        self.observation[0*self.get_width() + self.resources["wood"]]   = 1.0
        self.observation[1*self.get_width() + self.resources["brick"]]  = 1.0
        self.observation[2*self.get_width() + self.resources["wool"]]   = 1.0
        self.observation[3*self.get_width() + self.resources["crop"]]   = 1.0
        self.observation[4*self.get_width() + self.resources["ore"]]    = 1.0


        self.observation[5*self.get_width() + self.items["village"]]    = 1.0
        self.observation[6*self.get_width() + self.items["city"]]       = 1.0
        self.observation[7*self.get_width() + self.items["up city"]]    = 1.0
        self.observation[8*self.get_width() + self.items["road"]]       = 1.0
        self.observation[9*self.get_width() + self.items["knight"]]     = 1.0


    def __compute_score(self):
        result = 0
        result+= self.items["village"]*self.points["village"]
        result+= self.items["city"]*self.points["city"]
        result+= self.items["up city"]*self.points["up city"]
        result+= self.items["road"]*self.points["road"]
        result+= self.items["knight"]*self.points["knight"]

        return result

    def __get_costs(self, action):
        if action == 0:
            costs = self.costs["village"]
        elif action == 1:
            costs = self.costs["city"]
        elif action == 2:
            costs = self.costs["up city"]
        elif action == 3:
            costs = self.costs["road"]
        elif action == 4:
            costs = self.costs["knight"]
        else:
            costs = self.costs["pass"]

        return costs

    def __is_legal_action(self, action):

        costs = self.__get_costs(action)

        result = True

        if self.resources["wood"] < costs[0]:
            result = False
        if self.resources["brick"] < costs[1]:
            result = False
        if self.resources["wool"] < costs[2]:
            result = False
        if self.resources["crop"] < costs[3]:
            result = False
        if self.resources["ore"] < costs[4]:
            result = False

        if (action == 1) and (self.items["village"] <= 0):
            result = False
        if (action == 2) and (self.items["city"] <= 0):
            result = False

        return result

    def __execute_action(self, action):
        costs = self.__get_costs(action)

        self.resources["wood"]-=     costs[0]
        self.resources["brick"]-=    costs[1]
        self.resources["wool"]-=     costs[2]
        self.resources["crop"]-=     costs[3]
        self.resources["ore"]-=      costs[4]


        if action == 0:
            self.items["village"]+= 1
            points = self.points["village"]
        elif action == 1:
            self.items["village"]-= 1
            self.items["city"]+= 1
            points = self.points["city"]
        elif action == 2:
            self.items["city"]-= 1
            self.items["up city"]+= 1
            points = self.points["up city"]
        elif action == 3:
            self.items["road"]+= 1
            points = self.points["road"]
        elif action == 4:
            self.items["knight"]+= 1
            points = self.points["knight"]
        else:
            points = self.points["pass"]
            pass

        return points

    def __saturate_resources(self):
        max = self.width - 1
        min = 0

        if self.resources["wood"] > max:
            self.resources["wood"] = max

        if self.resources["brick"] > max:
            self.resources["brick"] = max

        if self.resources["wool"] > max:
            self.resources["wool"] = max

        if self.resources["crop"] > max:
            self.resources["crop"] = max

        if self.resources["ore"] > max:
            self.resources["ore"] = max



    def __saturate_items(self):
        max = self.width - 1

        if self.items["village"] > max:
            self.items["village"] = max

        if self.items["city"] > max:
            self.items["city"] = max

        if self.items["up city"] > max:
            self.items["up city"] = max

        if self.items["road"] > max:
            self.items["road"] = max

        if self.items["knight"] > max:
            self.items["knight"] = max

    def __get_card(self):
        num = random.randint(0, 4)

        if num == 0:
            result = "wood"
        elif num == 1:
            result = "brick"
        elif num == 2:
            result = "wool"
        elif num == 3:
            result = "crop"
        else:
            result = "ore"

        return result

    def __random_costs(self, count):
        result = [0, 0, 0, 0, 0]

        for i in range(0, count):
            idx = random.randint(0, 4)
            result[idx]+= 1

        return result
