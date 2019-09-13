import json


class RLExperiment:

    def __init__(self, env, config_dir):
        self.env            = env
        self.config_dir     = config_dir

    def process(self):

        with open(self.config_dir + "experiment_config.json") as json_file:
            config_json = json.load(json_file)

        verbose                 = bool(config_json["verbose"])

        trials_count            = int(config_json["trials_count"])
        training_games_count    = int(config_json["training_games_count"])
        testing_games_count     = int(config_json["testing_games_count"])


        #init DQN agent parameters
        gamma               = float(config_json["gamma"])
        replay_buffer_size  = int(config_json["replay_buffer_size"])
        epsilon_training    = float(config_json["epsilon_training"])
        epsilon_testing     = float(config_json["epsilon_testing"])
        epsilon_decay       = float(config_json["epsilon_decay"])


        for trial in range(0, trials_count):

            network_path = self.config_dir + "network/"
            agent = libs.libs_agent.agent_dqna.DQNAgent(env, network_path, gamma, replay_buffer_size, epsilon_training, epsilon_testing, epsilon_decay)


            training_games_count = 500
            while env.get_games_count() < training_games_count:
                agent.main()

                #print training progress %, ane score, every 256th iterations
                if verbose:
                    if env.get_iterations()%256 == 0:
                        env._print()
                        env.render()


                if env.get_iterations()%1000 == 0:
                    #TODO add results to log

                if env.get_iterations()%50000 == 0:
                    print("SAVING network")
                    agent.save(network_path + "trained/")


            agent.save(network_path + "trained/")
            agent.load(network_path + "trained/")

            #reset score
            env.reset_score()

            #choose only the best action
            agent.run_best_enable()


            #process testing games
            while env.get_games_count() < training_games_count + testing_games_to_play:
                agent.main()
