import mylibs.agent
import numpy



class QAgent(mylibs.agent.Agent):

    def __init__(self, env):
        mylibs.agent.Agent.__init__(self, env)

        self.gamma = 0.9
        self.alpha = 0.2

        self.epsilon_training = 0.1
        self.epsilon_testing  = 0.01

        self.state = 0
        self.state_prev = self.state

        self.action = 0
        self.action_prev = self.action

        self.states_count = self.env.get_size()
        self.actions_count = self.env.get_actions_count()

        self.q_table = numpy.zeros((self.states_count, self.actions_count))


    def main(self):

        if self.is_run_best_enabled():
            epsilon = self.epsilon_testing
        else:
            epsilon = self.epsilon_training

        self.state_prev = self.state
        self.state = self.env.get_observation().argmax()

        self.action_prev = self.action
        self.action = self.select_action(self.q_table[self.state], epsilon)

        reward = self.env.get_reward()

        q_max = self.q_table[self.state].max()
        dif =  reward + self.gamma*q_max - self.q_table[self.state_prev][self.action_prev]

        self.q_table[self.state_prev][self.action_prev]+= self.alpha*dif

        self.env.do_action(self.action)

    def print_table(self):
        print(self.q_table)
