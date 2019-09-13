import libs.libs_env.blackbox.env_black_box as blackbox
import libs.libs_agent.agent_dqn
import numpy


class BlackBoxTrial:

    def __init__(self, bot_file_name, training_iterations, testing_iterations, seed = 0, verbose = False):

        self.verbose = verbose
        self.env = blackbox.EnvBlackBox(seed)

        #print environment info
        if (verbose):
            self.env.print_info()
            print("loading agent ", bot_file_name)

        #init DQN agent
        self.agent = libs.libs_agent.agent_dqn.DQNAgent(self.env, bot_file_name, 0.3, 0.05, 0.99999)

        #iterations count
        self.training_iterations    = training_iterations
        self.testing_iterations     = testing_iterations



    #process training
    def train(self):
        #train bot
        for iteration in range(0, self.training_iterations):
            self.agent.main()

            #print debug info
            if self.verbose:
                if iteration%1000 == 0:
                    print(iteration*100.0/self.training_iterations, self.env.get_score())
                    self.env._print()

    #process testing run
    def test(self):

        #reset score
        self.env.reset_score()
        self.env.reset()

        #choose only the best action
        self.agent.run_best_enable()

        #process testing iterations
        for iteration in range(0, self.testing_iterations):
            #process agent
            self.agent.main()

            if self.verbose:
                print("move=", self.env.get_move(), " score=", self.env.get_score())

    def get_score(self):
        return self.env.get_score()

    def get_size(self):
        return self.env.get_size()

    def show(self):
        self.env.show()
