import libs.libs_agent.agent_dqn
import libs.libs_agent.agent_dqna
import libs.libs_agent.agent_dqn_curiosity

import libs.libs_rysy_python.rysy as rysy


class AtariRLDqn:

    def __init__(self, env, network_path, agent_type = "dqn", ):
        self.env = env
        self.network_path = network_path


        #init DQN agent
        gamma = 0.99
        replay_buffer_size  = 8192
        epsilon_start    = 1.0
        epsilon_end     = 0.1
        epsilon_decay       = 0.99999
        curiosity_ratio     = 0.1

        self.training_games_to_play = 0

        if agent_type == "dqn":
            self.agent = libs.libs_agent.agent_dqn.DQNAgent(self.env, self.network_path + "network_config.json", gamma, replay_buffer_size, epsilon_start, epsilon_end, epsilon_decay)
        elif agent_type == "dqrn":
            self.agent = libs.libs_agent.agent_dqn.DQNAgent(self.env, self.network_path + "network_config.json", gamma, replay_buffer_size, epsilon_start, epsilon_end, epsilon_decay, True)
        elif agent_type == "dqna":
            self.agent = libs.libs_agent.agent_dqna.DQNAAgent(env, self.network_path, gamma, replay_buffer_size, epsilon_start, epsilon_end, epsilon_decay)
        elif agent_type == "curiosity":
            self.agent = libs.libs_agent.agent_dqn_curiosity.DQNCuriosityAgent(env, self.network_path, gamma, curiosity_ratio, epsilon_start, epsilon_end, epsilon_decay)



    def visualise(self):
        self.agent.load(self.network_path + "trained/")

        self.agent.run_best_enable()

        while True:
            self.agent.main()

            heat_map = self.agent.get_heatmap()
            self.env.set_heat_map(heat_map)
            self.env.render()
            self.env._print()

    def kernel_visualisation(self):
        self.agent.load(self.network_path + "trained/")
        self.agent.kernel_visualisation(self.network_path + "kernel_visualisation/")

    def activity_visualisation(self, path = ""):
        self.agent.load(self.network_path + "trained/")
        self.agent.activity_visualisation(self.network_path + "activity_visualisation/" + path)


    def train(self, training_games_to_play = 500):
        training_progress_log = rysy.Log(self.network_path + "progress_training.log")

        self.training_games_to_play = training_games_to_play
        #process training
        while self.env.get_games_count() < self.training_games_to_play:
            self.agent.main()

            #print training progress %, ane score, every 256th iterations
            if self.env.get_iterations()%256 == 0:
                str_progress = str(self.env.get_iterations()) + " "
                str_progress+= str(self.env.get_games_count()) + " "
                str_progress+= str(self.agent.get_epsilon_start()) + " "
                str_progress+= str(self.env.get_score()) + " "
                str_progress+= "\n"
                training_progress_log.put_string(str_progress)

                print("done = ", self.env.get_games_count()*100.0/self.training_games_to_play, "%", " eps = ", self.agent.get_epsilon_start(), " iterations = ",  self.env.get_iterations(), " score = ",  self.env.get_score())

            if self.env.get_iterations()%50000 == 0:
                print("SAVING network")
                self.agent.save(self.network_path + "trained/")

        self.agent.save(self.network_path + "trained/")


    def test(self, testing_games_to_play = 100):
        testing_progress_log = rysy.Log(self.network_path + "progress_testing.log")

        self.agent.load(self.network_path + "trained/")

        #reset score
        self.env.reset_score()

        #choose only the best action
        self.agent.run_best_enable()


        #process testing games
        while self.env.get_games_count() < testing_games_to_play + self.training_games_to_play:
            self.agent.main()

            if self.env.get_iterations()%256 == 0:
                str_progress = str(self.env.get_iterations()) + " "
                str_progress+= str(self.env.get_games_count() - + self.training_games_to_play) + " "
                str_progress+= str(self.agent.get_epsilon_start()) + " "
                str_progress+= str(self.env.get_score()) + " "
                str_progress+= "\n"
                testing_progress_log.put_string(str_progress)


        print("TESTING SCORE =", self.env.get_score())

        return self.env.get_score()
