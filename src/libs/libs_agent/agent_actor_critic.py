import numpy
import random
import math
import libs.libs_agent.agent as libs_agent

from libs.libs_rysy_python.rysy import *


class ActorCritic(libs_agent.Agent):
    def __init__(self, env, network_config_path, gamma, replay_buffer_size, epsilon_start = 1.0, epsilon_end = 0.1, epsilon_decay = 0.99999):

        #init parent class
        libs_agent.Agent.__init__(self, env)

        state_shape  = Shape(self.env.get_width(), self.env.get_height(), self.env.get_depth()*self.env.get_time())

        self.model_actor   = CNN(network_config_path + "actor_config.json", state_shape, Shape(1, 1, self.env.get_actions_count()))
        self.model_critic  = CNN(network_config_path + "critic_config.json", state_shape, Shape(1, 1, 1))

        #init probabilities of choosing random action
        #different for training and testing
        self.epsilon_start      = 0.0

        self.gamma = gamma

        self.replay_buffer_size = replay_buffer_size
        self.replay_buffer = []

        self.model_actor._print()
        self.model_critic._print()


    def main(self):
        #self.epsilon_start = self.epsilon_end
        #epsilon = self.epsilon_end


        state           = self.env.get_observation()
        state_vector    = VectorFloat(state)    #convert to C++ vector
        actor_output    = VectorFloat(self.env.get_actions_count())
        critic_output   = VectorFloat(1)


        self.model_critic.forward(critic_output, state_vector)

        #obtain actor policy output
        self.model_actor.forward(actor_output, state_vector)
        policy_probs = self.__softmax(actor_output)


        actions = list(range(self.env.get_actions_count()))
        self.action = numpy.random.choice(actions, p = policy_probs)

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
                "policy_probs"  : policy_probs,
                "action"        : self.action,
                "reward"        : self.reward,
                "critic_output" : critic_output,
                "terminal"      : self.env.is_done()
            }
            self.replay_buffer.append(buffer_item)
        else:
            #compute buffer
            for n in reversed(range(self.replay_buffer_size-1)):

                #choose zero q if current state is terminal
                if self.replay_buffer[n]["terminal"] == True:
                    gamma = 0.0
                else:
                    gamma = self.gamma

                self.replay_buffer[n]["critic_output"][0] = self.replay_buffer[n]["reward"] + gamma*self.replay_buffer[n+1]["critic_output"][0]


            #shuffle items order
            indicies = numpy.arange(self.replay_buffer_size)
            numpy.random.shuffle(indicies)

            #train critic
            self.model_critic.set_training_mode()
            for i in range(len(indicies)):
                idx = indicies[i]
                self.model_critic.train(self.replay_buffer[idx]["critic_output"], self.replay_buffer[idx]["state"])
            self.model_critic.unset_training_mode()

            #train actor
            self.model_actor.set_training_mode()
            for i in range(len(indicies)):
                idx = indicies[i]

                action_idx      = self.replay_buffer[idx]["action"]
                policy_probs    = self.replay_buffer[idx]["policy_probs"]
                d_softmax       = self.__softmax_grad(policy_probs, action_idx)
                d_log           = d_softmax/policy_probs[action_idx]

                gradient    = VectorFloat(self.env.get_actions_count())

                q = self.replay_buffer[idx]["critic_output"][0]
                for j in range(0, gradient.size()):
                    gradient[j] = q*d_log[j]

                state  = self.replay_buffer[idx]["state"]

                self.model_actor.forward(actor_output, state)
                self.model_actor.train_from_gradient(gradient)


            self.model_actor.unset_training_mode()

            #clear buffer
            self.replay_buffer = []


    def __clamp(self, value, min, max):
        if value < min:
            value = min

        if value > max:
            value = max

        return value

    """
    compute softmax
    """
    def __softmax(self, input):
        result    = numpy.zeros(self.env.get_actions_count())

        max_value = numpy.max(input)

        for i in range(input.size()):
            result[i] = numpy.exp(input[i] - max_value)

        result = result/numpy.sum(result)

        return result

    """
    compute softmax gradient with respect to selected class
    """
    def __softmax_grad(self, softmax, selected):
        result = numpy.zeros(len(softmax))

        for i in range(0, len(softmax)):
            if i == selected:
                result[i] = softmax[selected]*(1.0 - softmax[i])
            else:
                result[i] = softmax[selected]*(-softmax[i])

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
        self.model_critic.load_weights(file_name_prefix  + "critic_trained/")

    def get_epsilon_start(self):
        return self.epsilon_start

    def get_heatmap(self):
        state = self.env.get_observation()
        state_vector = VectorFloat(state)

        return self.model.heatmap_compute(state_vector)
