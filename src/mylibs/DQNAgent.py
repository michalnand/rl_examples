import mylibs.agent
import numpy


import libs.libs_dqn_python_cpu.dqn as dqn
#import libs.libs_dqn_python_ubuntu.dqn as dqn
#import libs.libs_dqn_python.dqn as dqn

class DQNAgent(mylibs.agent.Agent):

    def __init__(self, env, network_config_file, epsilon_training = 0.2, epsilon_testing = 0.01, epsilon_decay = 1.0):

        mylibs.agent.Agent.__init__(self, env)

        state_geometry = dqn.sGeometry()

        state_geometry.w = self.env.get_width()
        state_geometry.h = self.env.get_height()
        state_geometry.d = self.env.get_depth()

        actions_count = self.env.get_actions_count()

        self.deep_q_network = dqn.DQN(network_config_file, state_geometry, actions_count)

        self.epsilon_training   = epsilon_training
        self.epsilon_testing    = epsilon_testing
        self.epsilon_decay      = epsilon_decay

    def main(self):

        if self.is_run_best_enabled():
            epsilon = self.epsilon_testing
        else:
            epsilon = self.epsilon_training
            self.epsilon_training*= self.epsilon_decay

        state = self.env.get_observation()
        state_vector = dqn.VectorFloat(self.env.get_size())
        for i in range(0, state_vector.size()):
            state_vector[i] = state[i]

        self.deep_q_network.compute_q_values(state_vector)
        q_values = self.deep_q_network.get_q_values()

        self.action = self.select_action(q_values, epsilon)

        self.env.do_action(self.action)

        self.reward = self.env.get_reward()

        if self.env.is_done():
            self.deep_q_network.add_final(state_vector, q_values, self.action, self.reward)
        else:
            self.deep_q_network.add(state_vector, q_values, self.action, self.reward)

        if self.deep_q_network.is_full() and self.is_run_best_enabled() == False:
            self.deep_q_network.learn()

    def save(self, file_name_prefix):
        self.deep_q_network.save(file_name_prefix)

    def load(self, file_name):
        self.deep_q_network.load_weights(file_name)
