import numpy
import time
import random


class BlackBoxLayer():
    def __init__(self, input_size, output_size, seed = 0):

        self.random_num = seed

        self.w = numpy.zeros((output_size, input_size))

        for j in range(0, output_size):
            for i in range(0, input_size):
                self.w[j][i] = 0.4*self.__rnd()

        self.output = numpy.zeros(output_size)
        self.reset()


    def reset(self):
        for i in range(0, len(self.output)):
            self.output[i] = 0.0

    def process(self, input):
        self.output = numpy.dot(self.w, input)
        self.output = numpy.tanh(self.output)

        return self.output

    def __rnd(self):
        return (self.__random()/32768.0 - 0.5)*2.0

    def __random(self):
        self.random_num = 1103515245*self.random_num + 12345
        return self.random_num%32768

class BlackBoxTransformation():

    def __init__(self, input_size, output_size, layers_count = 2, seed = 0):

        self.layers = []
        for layer in range(0, layers_count):
            layer_input_size  = input_size
            layer_output_size = input_size

            if layer == layers_count-1:
                layer_output_size = output_size

            layer_seed = layer + seed*(layer + 1)

            self.layers.append(BlackBoxLayer(layer_input_size, layer_output_size, layer_seed))

        self.reset()

    def reset(self):
        for i in range(0, len(self.layers)):
            self.layers[i].reset()

    def process(self, input):
        self.layer_output = numpy.copy(input)
        for i in range(0, len(self.layers)):
            self.layer_output = self.layers[i].process(self.layer_output)

        return self.layer_output



class BlackBox():

    def __init__(self, seed = 0):
        self.random_num = seed

        self.features_count = 32 + self.__random()%(32+1)
        self.actions_count  = 4  + self.__random()%(6+1)

        self.observation    = numpy.zeros(self.get_features_count())

        input_size  = self.features_count + self.actions_count
        output_size = self.features_count + 2
        self.transformation = BlackBoxTransformation(input_size, output_size, 3, self.__random())

        self.new_game   = False
        self.iterations = 0

        self.score = 0
        self.reward = 0

        self.init_new_game()

    def get_features_count(self):
        return self.features_count

    def get_actions_count(self):
        return self.actions_count

    def get_score(self):
        return self.score

    def reset_score(self):
        self.score = 0

    def get_reward(self):
        return self.reward

    def is_new_game(self):
        return self.new_game

    def get_iterations(self):
        return self.iterations

    def get_observation(self):
        return self.observation

    def do_action(self, action):

        self.action_vector = numpy.zeros(self.actions_count)
        self.action_vector[action%self.actions_count] = 1.0
        self.input_vector = numpy.concatenate((self.observation, self.action_vector))


        raw_output_vector = self.transformation.process(self.input_vector)

        ptr = 0
        for i in range(0, self.get_features_count()):
            self.observation[i] = raw_output_vector[ptr]
            ptr+= 1


        reward = raw_output_vector[ptr+0]
        reward+= -0.2

        shrink = 0.5
        if numpy.abs(reward) < shrink:
            reward = 0

        if reward > 1.0:
            reward = 1.0

        if reward < -1.0:
            reward = -1.0

        self.reward = reward

        game_end = numpy.abs(raw_output_vector[ptr+1])


        '''
        print("input_vector ", self.input_vector)
        print("raw output vector ", raw_output_vector)
        print("observation ", self.observation)
        print("reward ", self.reward)
        print("\n\n\n")
        '''

        self.new_game = False
        if (game_end > 0.99):
            self.init_new_game()

        self.score+= self.reward
        self.iterations+= 1

    def init_new_game(self):
        self.transformation.reset()
        self.input_vector = numpy.zeros(self.features_count + self.actions_count)
        self.observation    = numpy.zeros(self.get_features_count())

        self.new_game = True

    def __normalize_observation(self):
        min = numpy.min(self.observation)
        max = numpy.max(self.observation)

        k = 0.0
        if max > min:
            k = (1.0 - 0.0)/(max - min)
        q = 1.0 - k*max

        self.observation = k*self.observation + q


    def __random(self):
        self.random_num = 1103515245*self.random_num + 12345
        return self.random_num%32768
