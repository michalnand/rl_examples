"""
 deep Q network (DQN) reinforcement learning agent
 agent is using deep neural network and experience replay to learn Q(s, a) values

 parameters
 network_config_file_name - DQN neural network architecture
 epsilon_start - probability of choosing random action during training
 epsilon_end  - probability of choosing random action during testing or final value
 epsilon_decay - dacay of epsilon during training
"""

import numpy
import random
import math
import libs.libs_agent.agent as libs_agent

from libs.libs_rysy_python.rysy import *


#deep Q network agent
class DQNAgent(libs_agent.Agent):
    def __init__(self, env, network_config_file_name, gamma, replay_buffer_size, epsilon_start = 0.2, epsilon_end = 0.01, epsilon_decay = 1.0, rnn_model = False):

        #init parent class
        libs_agent.Agent.__init__(self, env)


        #this is true magic of deep RL, deep Q network
        if rnn_model:
                state_shape = Shape(self.env.get_width(), self.env.get_height(), self.env.get_depth(), self.env.get_time())
                self.deep_q_network = DQRN(state_shape, self.env.get_actions_count(), gamma, replay_buffer_size, network_config_file_name)
        else:
                state_shape = Shape(self.env.get_width(), self.env.get_height(), self.env.get_depth()*self.env.get_time())
                self.deep_q_network = DQN(state_shape, self.env.get_actions_count(), gamma, replay_buffer_size, network_config_file_name)

        self.deep_q_network._print()

        #init probabilities of choosing random action
        #different for training and testing
        self.epsilon_start   = epsilon_start
        self.epsilon_end    = epsilon_end
        self.epsilon_decay      = epsilon_decay

        self.state_vector = VectorFloat(self.env.get_size())
        for i in range(0, self.state_vector.size()):
            self.state_vector[i] = random.random()

        self.zero_q_values_counter = 0

    def main(self):

        #choose correct epsilon - check if testing or training mode
        if self.is_run_best_enabled():
            epsilon = self.epsilon_end
        else:
            epsilon = self.epsilon_start
            if self.epsilon_start > self.epsilon_end:
                self.epsilon_start*= self.epsilon_decay

        state = self.env.get_observation()
        self.state_vector = VectorFloat(state)


        q_values = self.deep_q_network.forward(self.state_vector)

        q_check_result = self.check_q_values(q_values)
        if q_check_result != 0:
            return q_check_result

        #select action using q_values from NN and epsilon
        self.action = self.select_action(q_values, epsilon)

        #execute action
        self.env.do_action(self.action)

        #obtain reward
        self.reward = self.env.get_reward()

        #add state, q_values and reward into experience replay

        #if it is terminal state (game end) add it by calling add_final()
        terminal = self.env.is_done()
        self.deep_q_network.add(self.state_vector, q_values, self.action, self.reward, terminal)


        #if experience replay is full process training - but only when agent in training mode
        if self.deep_q_network.is_full() and self.is_run_best_enabled() == False:
            self.deep_q_network.train()

        return 0

    """
        save agent neural network into specified dir
    """
    def save(self, file_name_prefix):
        self.deep_q_network.save(file_name_prefix)

    """
        load agent neural network from specified file
    """
    def load(self, file_name):
        self.deep_q_network.load_weights(file_name)

    def get_epsilon_start(self):
        return self.epsilon_start


    def check_q_values(self, q_values):
        for i in range(0, len(q_values)):
            if math.isnan(q_values[i]):
                 return -1

        zero_cnt = 0
        for i in range(0, len(q_values)):
            if q_values[i] == 0.0:
                zero_cnt+= 1

        if zero_cnt == len(q_values):
            self.zero_q_values_counter+= 1
        else:
            self.zero_q_values_counter = 0

        if self.zero_q_values_counter > 1024:
            return -2

        return 0

    def get_heatmap(self):
        state = self.env.get_observation()
        state_vector = VectorFloat(state)

        return self.deep_q_network.heatmap_compute(state_vector)
