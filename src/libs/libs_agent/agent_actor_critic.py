"""
 ActorCriticreinforcement learning agent
 agent is using deep neural network

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


epsilon = 0.001

class ActorCritic(libs_agent.Agent):
    def __init__(self, env, network_config_path, gamma, replay_buffer_size, epsilon_start = 1.0, epsilon_end = 0.1, epsilon_decay = 0.99999):

        #init parent class
        libs_agent.Agent.__init__(self, env)

        state_shape  = Shape(self.env.get_width(), self.env.get_height(), self.env.get_depth()*self.env.get_time())

        self.model_actor   = CNN(network_config_path + "actor_config.json", state_shape, Shape(1, 1, self.env.get_actions_count()))
        self.model_critic  = CNN(network_config_path + "critic_config.json", state_shape, Shape(1, 1, 1))

        #init probabilities of choosing random action
        #different for training and testing
        self.epsilon_start      = epsilon_start
        self.epsilon_end        = epsilon_end
        self.epsilon_decay      = epsilon_decay

        self.gamma = gamma

        self.replay_buffer_size = replay_buffer_size
        self.replay_buffer = []

        self.model_actor._print()
        self.model_critic._print()

    def main(self):
        self.epsilon_start = self.epsilon_end
        epsilon = self.epsilon_end

        '''
        if self.is_run_best_enabled():
            epsilon = self.epsilon_end
        else:
            epsilon = self.epsilon_start
            if self.epsilon_start > self.epsilon_end:
                self.epsilon_start*= self.epsilon_decay
        '''


        state           = self.env.get_observation()
        state_vector    = VectorFloat(state)    #convert to C++ vector
        actor_output    = VectorFloat(self.env.get_actions_count())
        critic_output   = VectorFloat(1)

        #obtain actor policy output
        self.model_actor.forward(actor_output, state_vector)

        #obtain critic output
        self.model_critic.forward(critic_output, state_vector)

        #select action using q_values from NN and epsilon
        self.action = self.select_action(actor_output, epsilon)

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
                "state"         : state_vector,
                "actor_output"  : actor_output,
                "critic_output" : critic_output,
                "action"        : self.action,
                "reward"        : self.reward,
                "terminal"      : self.env.is_done()
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

                #critic learning
                q      = self.replay_buffer[n]["reward"] + gamma*self.replay_buffer[n + 1]["critic_output"][0]
                self.replay_buffer[n]["critic_output"][0] = self.__clamp(q, -10.0, 10.0)

                policy_probs = self.__vector_to_probs(self.replay_buffer[n]["actor_output"])

                #actor learning
                action_id = self.replay_buffer[n]["action"]
                self.replay_buffer[n]["actor_output"][action_id] = q*1.0/(policy_probs[action_id] + epsilon)

                #for action_id in range(0, policy_probs.size()):
                #    self.replay_buffer[n]["actor_output"][action_id] = q*1.0/(policy_probs[action_id] + epsilon)

                #clamp Q values into range <-10, 10> to prevent divergence
                for action in range(self.env.get_actions_count()):
                    self.replay_buffer[n]["actor_output"][action] = self.__clamp(self.replay_buffer[n]["actor_output"][action], -10.0, 10.0)



            #shuffle items order
            indicies = numpy.arange(self.replay_buffer_size)
            numpy.random.shuffle(indicies)

            #train actor
            self.model_actor.set_training_mode()
            for i in range(len(indicies)):
                idx = indicies[i]

                state  = self.replay_buffer[idx]["state"]
                target = self.replay_buffer[idx]["actor_output"]

                #fit network
                self.model_actor.train(target, state)
            self.model_actor.unset_training_mode()

            #train critic
            self.model_critic.set_training_mode()
            for i in range(len(indicies)):
                idx = indicies[i]

                state  = self.replay_buffer[idx]["state"]
                target = self.replay_buffer[idx]["critic_output"]

                #fit network
                self.model_critic.train(target, state)
            self.model_critic.unset_training_mode()

            #clear buffer
            self.replay_buffer = []


    def __clamp(self, value, min, max):
        if value < min:
            value = min

        if value > max:
            value = max

        return value

    def __vector_to_probs(self, input):
        result    = VectorFloat(self.env.get_actions_count())

        sum = 0.0
        for i in range(input.size()):
            result[i] = numpy.exp(input[i])
            sum+= result[i]

        for i in range(input.size()):
            result[i]/= sum

        return result


    """
        save agent neural network into specified dir
    """
    def save(self, file_name_prefix):
        print("saving to", file_name_prefix)
        self.model_actor.save(file_name_prefix + "actor_trained/")
        self.model_critic.save(file_name_prefix + "critic_trained/")

    """
        load agent neural network from specified file
    """
    def load(self, file_name_prefix):
        print("loading weights from ", file_name_prefix)
        self.model_actor.load_weights(file_name_prefix  + "actor_trained/")
        self.model_critic.load_weights(file_name_prefix + "critic_trained/")

    def get_epsilon_start(self):
        return self.epsilon_start

    def get_heatmap(self):
        state = self.env.get_observation()
        state_vector = VectorFloat(state)

        return self.model_actor.heatmap_compute(state_vector)
