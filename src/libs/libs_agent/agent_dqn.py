"""
 deep Q network (DQN) reinforcement learning agent
 agent is using deep neural network and experience replay to learn Q(s, a) values

 parameters
 agent_config_path - config dir
 epsilon_start - probability of choosing random action during training
 epsilon_end  - probability of choosing random action during testing or final value
 epsilon_decay - dacay of epsilon during training
"""

import numpy
import random
import math
import json
import libs.libs_agent.agent as libs_agent

from libs.libs_rysy_python.rysy import *




#deep Q network agent
class DQNAgent(libs_agent.Agent):
    def __init__(self, env, agent_config_path):

        #init parent class
        libs_agent.Agent.__init__(self, env)

        json_file = open(agent_config_path + "agent_config.json")
        json_data = json.load(json_file)

        self.replay_buffer_size = int(json_data["replay_buffer_size"])
        self.gamma              = float(json_data["gamma"])
        self.epsilon_start      = float(json_data["epsilon_start"])
        self.epsilon_end        = float(json_data["epsilon_end"])
        self.epsilon_decay      = float(json_data["epsilon_decay"])
        self.replay_buffer      = []


        state_shape  = Shape(self.env.get_width(), self.env.get_height(), self.env.get_depth()*self.env.get_time())
        output_shape = Shape(1, 1, self.env.get_actions_count())
        self.model   = CNN(agent_config_path + "network_config.json", state_shape, output_shape)


        self.model._print()

    def main(self):
         #choose correct epsilon - check if testing or training mode
        if self.is_run_best_enabled():
            epsilon = self.epsilon_end
        else:
            epsilon = self.epsilon_start
            if self.epsilon_start > self.epsilon_end:
                self.epsilon_start*= self.epsilon_decay

        state           = self.env.get_observation()
        state_vector    = VectorFloat(state)    #convert to C++ vector
        q_values        = VectorFloat(self.env.get_actions_count())

        #obtain Q-values from state
        self.model.forward(q_values, state_vector)

        #select action using q_values from NN and epsilon
        self.action = self.select_action(q_values, epsilon)

        #execute action
        self.env.do_action(self.action)

        #obtain reward
        self.reward = self.env.get_reward()

        if self.is_run_best_enabled() == True:
            return

        #add to experience replay buffer
        #- state, q_values, reward, terminal state flag
        if len(self.replay_buffer) < self.replay_buffer_size:
            buffer_item  = {
                "state"        : state_vector,
                "q_values"     : q_values,
                "action"       : self.action,
                "reward"       : self.reward,
                "terminal"     : self.env.is_done()
            }
            self.replay_buffer.append(buffer_item)
        else:
            #compute buffer Q values, using Q learning
            for n in reversed(range(self.replay_buffer_size-1)):

                #choose zero gamme if current state is terminal
                if self.replay_buffer[n]["terminal"] == True:
                    gamma = 0.0
                else:
                    gamma = self.gamma

                action_id = self.replay_buffer[n]["action"]


                #Q-learning : Q(s[n], a[n]) = R[n] + gamma*max(Q(s[n+1]))
                q_next = max(self.replay_buffer[n+1]["q_values"])
                self.replay_buffer[n]["q_values"][action_id] = self.replay_buffer[n]["reward"] + gamma*max(self.replay_buffer[n+1]["q_values"])

                #clamp Q values into range <-10, 10> to prevent divergence
                for action in range(self.env.get_actions_count()):
                    self.replay_buffer[n]["q_values"][action] = self.__clamp(self.replay_buffer[n]["q_values"][action], -10.0, 10.0)


            '''
            common supervised training
                we have in/out pairs :
                    input         = self.replay_buffer[n]["state"]
                    target output = self.replay_buffer[n]["q_values"]
            '''

            #shuffle items order
            indicies = numpy.arange(self.replay_buffer_size)
            numpy.random.shuffle(indicies)

            self.model.set_training_mode()

            for i in range(len(indicies)):
                #choose random item, to break correlations
                idx = indicies[i]

                state           = self.replay_buffer[idx]["state"]
                target_q_values = self.replay_buffer[idx]["q_values"]

                #fit network
                self.model.train(target_q_values, state)

            self.model.unset_training_mode()

            #clear buffer
            self.replay_buffer = []

        return 0


    def __clamp(self, value, min, max):
        if value < min:
            value = min

        if value > max:
            value = max

        return value


    """
        save agent neural network into specified dir
    """
    def save(self, file_name_prefix):
        self.model.save(file_name_prefix + "trained/")

    """
        load agent neural network from specified file
    """
    def load(self, file_name_prefix):
        print("loading weights from ", file_name_prefix + "trained/")
        self.model.load_weights(file_name_prefix + "trained/")

    def get_epsilon_start(self):
        return self.epsilon_start

    def get_heatmap(self):
        state = self.env.get_observation()
        state_vector = VectorFloat(state)

        return self.model.heatmap_compute(state_vector)
