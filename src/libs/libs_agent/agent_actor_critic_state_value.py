import numpy
import random
import math
import libs.libs_agent.agent as libs_agent

from libs.libs_rysy_python.rysy import *


class ActorCriticStateValue(libs_agent.Agent):
    def __init__(self, env, network_config_path, gamma, replay_buffer_size, epsilon_start = 1.0, epsilon_end = 0.1, epsilon_decay = 0.99999):

        #init parent class
        libs_agent.Agent.__init__(self, env)

        state_shape  = Shape(self.env.get_width(), self.env.get_height(), self.env.get_depth()*self.env.get_time())

        self.actor_model   = CNN(network_config_path + "actor_network_config.json", state_shape, Shape(1, 1, self.env.get_actions_count()))
        self.critic_model   = CNN(network_config_path + "critic_network_config.json", state_shape, Shape(1, 1, 1))

        #init probabilities of choosing random action
        #different for training and testing
        self.epsilon_start      = epsilon_end
        self.epsilon_end        = epsilon_end

        self.gamma = gamma

        self.replay_buffer_size = replay_buffer_size
        self.replay_buffer = []

        self.q_values = numpy.zeros(replay_buffer_size)

        self.actor_model._print()
        self.critic_model._print()


    def main(self):
        #self.epsilon_start = self.epsilon_end
        #epsilon = self.epsilon_end

        state           = self.env.get_observation()
        state_vector    = VectorFloat(state)    #convert to C++ vector
        actor_output    = VectorFloat(self.actor_model.get_output_shape().size())

        #critic_output   = VectorFloat(self.critic_model.get_output_shape().size())
        #self.critic_model.forward(critic_output, state_vector)

        #obtain actor policy output
        self.actor_model.forward(actor_output, state_vector)
        policy_probs = self.__softmax(actor_output)

        self.action = self.__select_action(policy_probs, 0.0)

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
                "action"        : int(self.action),
                "reward"        : self.reward,
                "q"             : 0.0,
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

                q_new                       = self.replay_buffer[n]["reward"] + gamma*self.replay_buffer[n+1]["q"]
                self.replay_buffer[n]["q"]  = q_new
                self.q_values[n] = q_new

            self.q_values = (self.q_values - self.q_values.mean())/self.q_values.std()

            #self.__train_critic()
            self.__train_actor()


            #clear buffer
            self.replay_buffer = []

    def __train_critic(self):

        #shuffle items order
        indicies = numpy.arange(self.replay_buffer_size)
        numpy.random.shuffle(indicies)

        critic_target = VectorFloat(self.critic_model.get_output_shape().size())
        self.critic_model.set_training_mode()

        for i in range(len(indicies)):
            idx                 = indicies[i]
            state               = self.replay_buffer[idx]["state"]
            critic_target[0]    = self.replay_buffer[idx]["q"]

            self.critic_model.train(critic_target, state)

        self.critic_model.unset_training_mode()

    def __train_actor(self):
        #shuffle items order
        indicies = numpy.arange(self.replay_buffer_size)
        numpy.random.shuffle(indicies)

        actor_output = VectorFloat(self.actor_model.get_output_shape().size())
        gradient     = VectorFloat(self.actor_model.get_output_shape().size())

        self.actor_model.set_training_mode()

        for i in range(len(indicies)):
            idx = indicies[i]

            state           = self.replay_buffer[idx]["state"]
            action_idx      = self.replay_buffer[idx]["action"]
            policy_probs    = self.replay_buffer[idx]["policy_probs"]

            d_softmax       = self.__softmax_grad(policy_probs, action_idx)
            d_entropy       = self.__entropy_grad(policy_probs)

            beta        = 0.001

            q = self.q_values[idx]

            for j in range(0, gradient.size()):
                gradient[j] = q*d_softmax[j]/policy_probs[action_idx]



            self.actor_model.forward(actor_output, state)

            for j in range(0, gradient.size()):
                gradient[j]+= (-1.0)*beta*actor_output[j]

            self.actor_model.train_from_gradient(gradient)

            if (i%64) == 0:
                print(q, " :\n")

                for j in range(0, gradient.size()):
                    print(round(actor_output[j], 4), end = " ")

                print(" : ")

                for j in range(0, gradient.size()):
                    print(round(policy_probs[j], 4), end = " ")

                print(" : ")
                for j in range(0, gradient.size()):
                    print(round(gradient[j], 4), end = " ")

                print("\n\n")

        self.actor_model.unset_training_mode()

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

        for i in range(len(input)):
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
                result[i] = softmax[selected]*(0.0 - softmax[i])

        return result

    def __entropy(self, probs):
        return -numpy.sum(probs * numpy.log2(probs))

    def __entropy_grad(self, probs):
        tmp = numpy.clip(probs, 0.00000001, 1.0)
        return -(numpy.log2(tmp) + 1.0/numpy.log(2.0))

    def __select_action(self, probs, random_action_prob):
        actions = list(range(self.env.get_actions_count()))
        result = numpy.random.choice(actions, p = probs)

        if numpy.random.rand() < random_action_prob:
            result = numpy.random.randint(self.env.get_actions_count())

        return result


    """
        save agent neural network into specified dir
    """
    def save(self, file_name_prefix):
        print("saving to", file_name_prefix)
        self.actor_model.save(file_name_prefix + "actor_trained/")
        self.critic_model.save(file_name_prefix + "critic_trained/")

    """
        load agent neural network from specified file
    """
    def load(self, file_name_prefix):
        print("loading weights from ", file_name_prefix)
        self.actor_model.load_weights(file_name_prefix  + "actor_trained/")
        self.critic_model.load_weights(file_name_prefix  + "critic_trained/")

    def get_epsilon_start(self):
        return self.epsilon_start

    def get_heatmap(self):
        state = self.env.get_observation()
        state_vector = VectorFloat(state)

        return self.actor_model.heatmap_compute(state_vector)
