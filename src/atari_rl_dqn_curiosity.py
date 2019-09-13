import libs.libs_agent.agent_dqn_curiosity
import libs.libs_rysy_python.rysy as rysy


class AtariRLDqnCuriosity:

    def __init__(self, env, network_path, curiosity_ratio = 0.1):
        self.env = env
        self.network_path = network_path


        #init DQN agent
        gamma = 0.99
        replay_buffer_size  = 8192
        epsilon_training    = 1.0
        epsilon_testing     = 0.1
        epsilon_decay       = 0.99999

        self.training_games_to_play = 0

        self.agent = libs.libs_agent.agent_dqn_curiosity.DQNCuriosityAgent(env, self.network_path, gamma, curiosity_ratio,replay_buffer_size, epsilon_training, epsilon_testing, epsilon_decay)



    def visualise(self):
        self.agent.load(self.network_path)

        self.agent.run_best_enable()

        while True:
            self.agent.main()
            self.env.render()
            self.env._print()



    def train(self, training_games_to_play = 500):
        training_progress_log = rysy.Log(self.network_path + "progress_training.log")
        icm_training_progress_log = rysy.Log(self.network_path + "icm_progress_training.log")

        self.training_games_to_play = training_games_to_play
        #process training
        while self.env.get_games_count() < self.training_games_to_play:
            self.agent.main()


            #print training progress %, ane score, every 256th iterations
            if self.env.get_iterations()%256 == 0:
                str_progress = str(self.env.get_iterations()) + " "
                str_progress+= str(self.env.get_games_count()) + " "
                str_progress+= str(self.agent.get_epsilon_training()) + " "
                str_progress+= str(self.env.get_score()) + " "
                str_progress+= "\n"
                training_progress_log.put_string(str_progress)

                print("done = ", self.env.get_games_count()*100.0/self.training_games_to_play, "%", " eps = ", self.agent.get_epsilon_training(), " iterations = ",  self.env.get_iterations(), " score = ",  self.env.get_score())


            if self.env.get_iterations()%256 == 0:
                str_icm_progress = str(self.env.get_iterations()) + " "
                str_icm_progress+= str(self.env.get_games_count()) + " "
                str_icm_progress+= str(self.agent.get_icm_result().inverse_loss) + " "
                str_icm_progress+= str(self.agent.get_icm_result().forward_loss) + " "
                str_icm_progress+= str(self.agent.get_icm_result().inverse_classification_success) + " "
                str_icm_progress+= "\n"
                icm_training_progress_log.put_string(str_icm_progress)

            if self.env.get_iterations()%50000== 0:
                print("SAVING network")
                self.agent.save(self.network_path)

        self.agent.save(self.network_path)


    def test(self, testing_games_to_play = 100):
        testing_progress_log = rysy.Log(self.network_path + "progress_testing.log")

        self.agent.load(self.network_path)

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
                str_progress+= str(self.agent.get_epsilon_training()) + " "
                str_progress+= str(self.env.get_score()) + " "
                str_progress+= "\n"
                testing_progress_log.put_string(str_progress)


        print("TESTING SCORE =", self.env.get_score())

        return self.env.get_score()
