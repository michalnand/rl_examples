import random
import numpy
import libs.libs_env.env


class Agent():
    def __init__(self, env):
        self.env = env
        self.run_best_disable()

    def main(self):
        action = random.randint(0, self.env.get_actions_count()-1)
        self.env.do_action(action)



    def run_best_enable(self):
        self.run_best_enabled = True

    def run_best_disable(self):
        self.run_best_enabled = False

    def is_run_best_enabled(self):
        return self.run_best_enabled



    def select_action(self, q_values, epsilon = 0.1):
        action = self.__argmax(q_values)

        r = numpy.random.uniform(0.0, 1.0)
        if r <= epsilon:
            action = random.randint(0, self.env.get_actions_count()-1)

        return action

    def __argmax(self, v):
        result = 0
        for i in range(0, len(v)):
            if v[i] > v[result]:
                result = i

        return result
