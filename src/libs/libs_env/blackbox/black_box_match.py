import libs.libs_env.blackbox.black_box_trial as bb_trial
import json

import numpy


class BlackBoxMatch:

    def __init__(self, config_file_name):
        f = open(config_file_name)

        self.json_config = json.load(f)

        self.trials_count        = self.json_config["trials count"]
        self.training_iterations = self.json_config["training iterations"]
        self.testing_iterations  = self.json_config["testing iterations"]
        self.verbose             = self.json_config["verbose"]
        self.seed                = self.json_config["seed"]



        self.print_info()

    def print_info(self):
        print("trials count", self.trials_count)
        print("training iterations", self.training_iterations)
        print("testing iterations", self.testing_iterations)
        print("verbose", self.verbose)
        print("seed", self.seed)
        print()

    def run(self):
        bots = self.json_config["bots"]

        self.results = numpy.zeros((len(bots), self.trials_count))
        self.results_summary = numpy.zeros(len(bots))

        for j in range(0, len(bots)):

            bot_name = bots[j]["name"]
            bot_file_name = bots[j]["file name"]

            for i in range(0, self.trials_count):
                trial = bb_trial.BlackBoxTrial(bot_file_name, self.training_iterations, self.testing_iterations, self.seed + i, self.verbose)
                trial.train()
                trial.test()

                print("bot ", bot_name, " match ", i, "score = ", trial.get_score())
                print("\n\n\n")

                self.results[j][i] = trial.get_score()
                self.results_summary[j]+= trial.get_score()

    def print_score(self):
        bots = self.json_config["bots"]

        for j in range(0, len(bots)):
            bot_name = bots[j]["name"]
            print(bot_name, "score = ", self.results_summary[j], " : ", end = "")

            for i in range(0, self.trials_count):
                print(self.results[j][i], end = " ")
            print()
            print()

        print()
        print("summary ")

        for j in range(0, len(bots)):
            bot_name = bots[j]["name"]
            print(bot_name, "score = ", self.results_summary[j])

    def save_score(self):

        file_name = self.json_config["result file name"]
        f = open(file_name, 'w')

        bots = self.json_config["bots"]

        for j in range(0, len(bots)):
            bot_name = bots[j]["name"]
            print(bot_name, "score = ", self.results_summary[j], " : ", end = "", file = f)

            for i in range(0, self.trials_count):
                print(self.results[j][i], end = " ", file = f)
            print(file = f)
            print(file = f)

        print(file = f)
        print("summary ", file = f)

        for j in range(0, len(bots)):
            bot_name = bots[j]["name"]
            print(bot_name, "score = ", self.results_summary[j], file = f)
