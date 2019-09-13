"""
 deep Q network (DQN) reinforcement learning agent
 agent is using deep neural network and experience replay to learn Q(s, a) values

 parameters
 network_config_file_name - DQN neural network architecture
 epsilon_training - probability of choosing random action during training
 epsilon_testing  - probability of choosing random action during testing
 epsilon_decay - dacay of epsilon during training
"""

import sys
sys.path.append("..") # Adds higher directory to python modules path.


import numpy
import agent as agent

#uncoment this if you have CUDA GPU
import libs.libs_dqn_python.dqn as libs_dqn

#uncoment this if you hvae CPU and Debian
#import libs.libs_dqn_python_cpu.dqn as libs_dqn

#uncoment this if you hvae CPU and Ubuntu
#import libs.libs_dqn_python_cpu_ubuntu.dqn as libs_dqn

#deep Q network agent
class DQNAgent(agent.Agent):
    def __init__(self, env, network_config_file_name, epsilon_training = 0.2, epsilon_testing = 0.01, epsilon_decay = 1.0):
        print("DQNAgent")
        
        #init parent class
        agent.Agent.__init__(self, env)

        state_geometry = libs_dqn.sGeometry()
        state_geometry.w = self.env.get_width()
        state_geometry.h = self.env.get_height()
        state_geometry.d = self.env.get_depth()

        print("loading network from ", network_config_file_name)

        #this is true magic of deep RL, deep Q network
        self.deep_q_network = libs_dqn.DQN(network_config_file_name, state_geometry, self.env.get_actions_count())

        #init probabilities of choosing random action
        #different for training and testing
        self.epsilon_training   = epsilon_training
        self.epsilon_testing    = epsilon_testing
        self.epsilon_decay      = epsilon_decay

    def main(self):

        #choose correct epsilon - check if testing or training mode
        if self.is_run_best_enabled():
            epsilon = self.epsilon_testing
        else:
            epsilon = self.epsilon_training
            if self.epsilon_training > self.epsilon_testing:
                self.epsilon_training*= self.epsilon_decay

        #obtain state
        state = self.env.get_observation()
        state_vector = libs_dqn.VectorFloat(self.env.get_size())
        for i in range(0, state_vector.size()):
            state_vector[i] = state[i]

        #obtain Q values using neural network
        self.deep_q_network.compute_q_values(state_vector)
        q_values = self.deep_q_network.get_q_values()

        #select action using q_values from NN and epsilon
        self.action = self.select_action(q_values, epsilon)

        #execute action
        self.env.do_action(self.action)

        #obtain reward
        self.reward = self.env.get_reward()

        #add state, q_values and reward into experience replay

        #if it is terminal state (game end) add it by calling add_final()
        if self.env.is_done():
            self.deep_q_network.add_final(state_vector, q_values, self.action, self.reward)
        #otherwise add as common state
        else:
            self.deep_q_network.add(state_vector, q_values, self.action, self.reward)

        #if experience replay is full process training - but only when agent in training mode
        if self.deep_q_network.is_full() and self.is_run_best_enabled() == False:
            self.deep_q_network.learn()
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

    def get_epsilon_training(self):
        return self.epsilon_training
