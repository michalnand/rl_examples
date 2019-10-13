import numpy
import random
import math
import json
import libs.libs_agent.agent as libs_agent

from libs.libs_rysy_python.rysy import *


class ActorCritic(libs_agent.Agent):
    def __init__(self, env, agent_config_path):

        #init parent class
        libs_agent.Agent.__init__(self, env)

        json_file = open(agent_config_path + "agent_config.json")
        json_data = json.load(json_file)

        self.replay_buffer_size = int(json_data["replay_buffer_size"])
        self.gamma              = float(json_data["gamma"])
        self.beta               = float(json_data["beta"])
        self.best_action_probability    = float(json_data["best_action_probability"])
        self.replay_buffer      = []

        state_shape  = Shape(self.env.get_width(), self.env.get_height(), self.env.get_depth()*self.env.get_time())
        self.model   = CNN(agent_config_path + "network_config.json", state_shape, Shape(1, 1, self.env.get_actions_count() + 1))

        self.model._print()


    def main(self):
        actions_count   = self.env.get_actions_count()

        state           = self.env.get_observation()
        state_vector    = VectorFloat(state)
        model_output    = VectorFloat(actions_count+1)

        self.model.forward(model_output, state_vector)
        actor_output, critic_output = self.__split_model_output(model_output)

        policy_probs    = self.__softmax(actor_output)

        if self.is_run_best_enabled():
            self.action     = self.__select_best_action(policy_probs, 1.0- self.best_action_probability)
        else:
            self.action     = self.__select_action(policy_probs)

        self.env.do_action(self.action)

        #obtain reward
        self.reward = self.env.get_reward()

        if self.is_run_best_enabled():
            return

        if len(self.replay_buffer) < self.replay_buffer_size:
            buffer_item  = {
                "state"         : state_vector,
                "reward"        : self.reward,
                "terminal"      : self.env.is_done(),

                "action"        : int(self.action),
                "actor_output"  : actor_output,
                "policy_probs"  : policy_probs,
                "critic_output" : critic_output
            }
            self.replay_buffer.append(buffer_item)
        else:
            #compute buffer

            critic_error_sum    = 0.0
            entropy_error_sum   = 0.0

            r = numpy.zeros(len(self.replay_buffer))
            g = numpy.zeros(len(self.replay_buffer))
            for n in reversed(range(self.replay_buffer_size-1)):
                #choose zero q if current state is terminal
                if self.replay_buffer[n]["terminal"] == True:
                    gamma = 0.0
                else:
                    gamma = self.gamma

                r[n] = self.replay_buffer[n]["reward"] + gamma*r[n + 1]
                g[n] = r[n] - self.replay_buffer[n]["critic_output"]

                critic_error_sum+=  g[n]**2.0
                entropy_error_sum+= self.__entropy(self.replay_buffer[n]["policy_probs"])

            print("\n\n")
            print("critic_error_sum = ", critic_error_sum/len(self.replay_buffer))
            print("entropy_error_sum = ", entropy_error_sum/len(self.replay_buffer))
            print("\n\n")

            indicies = numpy.arange(len(self.replay_buffer))
            numpy.random.shuffle(indicies)

            model_gradients    = VectorFloat(actions_count+1)

            self.model.set_training_mode()
            for j in range(len(indicies)):
                idx = indicies[j]

                state               = self.replay_buffer[idx]["state"]
                policy_probs        = self.replay_buffer[idx]["policy_probs"]
                action              = self.replay_buffer[idx]["action"]


                self.model.forward(model_output, state)
                actor_output, critic_output = self.__split_model_output(model_output)

                #actions_vector = numpy.zeros(actions_count)
                #actions_vector[action] = 1.0

                softmax_gradient   = self.__softmax_grad(policy_probs)
                gradient_actor     = g[idx]*(softmax_gradient[action] / policy_probs[action])

                policy_probs_current = self.__softmax(actor_output)
                gradient_entropy     = self.beta*numpy.dot(self.__softmax_grad(policy_probs_current), self.__entropy_grad(policy_probs_current))

                for i in range(0, actions_count):
                    model_gradients[i] = (gradient_actor[i] + gradient_entropy[i])

                model_gradients[actions_count] = r[idx] - critic_output

                self.model.train_from_gradient(model_gradients)

                if j%256 == 0:
                    print(gradient_actor, gradient_entropy, policy_probs)


            self.model.unset_training_mode()

            self.replay_buffer = []

        return 0

    def train_model(self, replay_buffer):
        #shuffle items order
        indicies = numpy.arange(len(replay_buffer))
        numpy.random.shuffle(indicies)

        actions_count = self.env.get_actions_count()
        critic_model_output     = VectorFloat(actions_count)

        #train model
        self.critic_model.set_training_mode()

        for i in range(len(indicies)):
            idx = indicies[i]

            state           = replay_buffer[idx]["state"]
            critic_output   = replay_buffer[idx]["critic_output"]
            self.critic_model.train(critic_output, state)

        self.critic_model.unset_training_mode()

        critic_error = 0.0
        for i in range(len(indicies)):
            idx = indicies[i]

            state           = replay_buffer[idx]["state"]
            critic_output   = replay_buffer[idx]["critic_output"]

            self.critic_model.forward(critic_model_output, state)

            for j in range(0, actions_count):
                dif = critic_output[j] - critic_model_output[j]
                critic_error+= dif**2.0


            '''
            beta            = 0.0001

            self.model.forward(model_output, state)



            logits_output, critic_output    = self.__split_output(model_output)
            policy_probs                    = self.__softmax(logits_output)
            action                          = self.__select_action(policy_probs, 0.0)


            softmax_grad                    = self.__softmax_grad(policy_probs)

            #actor critic magic starts here :

            #gradients for actor, policy network
            #da = Q(s)*dP(a|s)/P(a|s) + beta*dH(P(s))
            gradients_entropy   = beta*self.__entropy_grad(softmax_grad, policy_probs)

            gradients_policy    = critic_output*softmax_grad[action]*(numpy.log(policy_probs))*(-1.0)


            gradients       = VectorFloat(actions_count + 1)
            for j in range(0, actions_count):
                gradients[j] = 0 #gradients_policy[j] + gradients_entropy[j]

            #gradient for critic, use commin rms, dc = Q(s) - Q'(s; w)
            gradients[actions_count] = critic_target - critic_output

            critic_error+= (critic_target - critic_output)**2.0

            self.model.train_from_gradient(gradients)

            if (i%64) == 0:
                for j in range(0, model_output.size()):
                    print(round(gradients[j], 4), end = " ")

                print(" : ")

                for j in range(0, actions_count+1):
                    print(round(model_output[j], 4), end = " ")

                print(" : ")

                for j in range(0, actions_count):
                    print(round(policy_probs[j], 4), end = " ")

                print("\n\n")
            '''

        print("\n\ncritic_error = ", critic_error/len(indicies), "\n\n\n")

    def __split_model_output(self, model_output):
        actions_count = len(model_output) - 1

        actor_output  = numpy.zeros(actions_count)
        for i in range(0, actions_count):
            actor_output[i] = model_output[i]

        critic_output = model_output[actions_count]
        return actor_output, critic_output

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

    def __entropy(self, probs):
        h = -numpy.sum(probs*numpy.log2(probs))
        return h

    def __softmax_grad(self, probs):
        s = probs.reshape(-1,1)
        return numpy.diagflat(s) - numpy.dot(s, s.T)

    def __entropy_grad(self, probs):
        eps = 0.0000001
        return -(numpy.log2(probs + eps) + 1.0/numpy.log(2.0))

    def __select_action(self, probs):
        actions = list(range(self.env.get_actions_count()))
        result = numpy.random.choice(actions, p = probs)
        return result

    def __select_best_action(self, probs, random_action_prob = 0.0):
        result = numpy.argmax(probs)
        if numpy.random.rand() < random_action_prob:
            result = numpy.random.randint(self.env.get_actions_count())

        return result


    """
        save agent neural network into specified dir
    """
    def save(self, file_name_prefix):
        print("saving to", file_name_prefix)
        self.model.save(file_name_prefix + "trained/")

    """
        load agent neural network from specified file
    """
    def load(self, file_name_prefix):
        print("loading weights from ", file_name_prefix)
        self.model.load_weights(file_name_prefix + "trained/")

    def get_epsilon_start(self):
        return self.epsilon_start

    def get_heatmap(self):
        state = self.env.get_observation()
        state_vector = VectorFloat(state)

        return self.actor_model.heatmap_compute(state_vector)
