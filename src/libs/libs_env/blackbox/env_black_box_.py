import scipy

import numpy
import time
import random

import libs.libs_env.env as libs_env



class BlackBoxLayer():
    def __init__(self, input_size, output_size, seed = 0):

        self.random_num = seed

        self.w = numpy.zeros((output_size, input_size))

        for j in range(0, output_size):
            for i in range(0, input_size):
                self.w[j][i] = 0.8*self.__rnd()

        self.output = numpy.zeros(output_size)
        self.reset()


    def reset(self):
        for i in range(0, len(self.output)):
            self.output[i] = 0.0

    def process(self, input):
        self.output = numpy.dot(self.w, input)
        self.output = numpy.tanh(self.output)

        return self.output

    def __random(self):
        self.random_num = 1103515245*self.random_num + 12345
        return self.random_num%32768

    def __rnd(self):
        return (self.__random()/32768.0 - 0.5)*2.0


class BlackBoxTransformation():

    def __init__(self, input_size, output_size, layers_count = 2, seed = 0):

        self.layers = []
        for layer in range(0, layers_count):
            layer_input_size  = input_size
            layer_output_size = input_size

            if layer == layers_count-1:
                layer_output_size = output_size

            layer_seed = seed*(layers_count + 1)

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






class EnvBlackBox(libs_env.Env):

    def __init__(self, seed = 0):
        #init parent class -> environment interface
        libs_env.Env.__init__(self)

        prime1 = 1001
        prime2 = 1007

        self.features_count = 8 + (seed*prime1)%24
        self.actions_count  = 4 + (seed*prime2)%5

        #state dimensions
        self.width  = 1
        self.height = 1
        self.depth  = self.features_count

        #init state, as 1D vector (tensor with size depth*height*width)
        self.observation    = numpy.zeros(self.get_size())


        input_size  = self.features_count + self.actions_count
        output_size = self.features_count + 1
        self.transformation = BlackBoxTransformation(input_size, output_size, 3, seed)

        self.iterations = 0
        self.reset()


    def reset(self):
        self.transformation.reset()
        self.input_vector = numpy.zeros(self.features_count + self.actions_count)
        self.observation    = numpy.zeros(self.get_size())

    def _print(self):
        print(self.get_move(), self.get_score(), self.get_observation(), self.reward)


    def do_action(self, action):

        self.set_no_terminal_state()

        self.action_vector = numpy.zeros(self.actions_count)
        self.action_vector[action%self.actions_count] = 1.0
        self.input_vector = numpy.concatenate((self.observation, self.action_vector))


        raw_output_vector = self.transformation.process(self.input_vector)


        ptr = 0
        for i in range(0, self.get_size()):
            self.observation[i] = raw_output_vector[ptr]
            ptr+= 1

        reward = raw_output_vector[ptr]
        shrink = 0.5

        if numpy.abs(reward) < shrink:
            reward = 0

        self.reward = reward


        '''
        print("input_vector ", self.input_vector)
        print("raw output vector ", raw_output_vector)
        print("observation ", self.observation)
        print("reward ", self.reward)
        print("\n\n\n")
        '''

        self.reward-= 0.1

        if (self.reward > 0.899):
            self.set_terminal_state()
            self.reset()
            #print("new game in ", self.get_iterations())

        self.next_move()
        self.iterations+= 1

    def get_iterations(self):
        return self.iterations
