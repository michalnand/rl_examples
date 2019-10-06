import libs.libs_agent.agent_dqn
import libs.libs_agent.agent_dqn_dueling
import libs.libs_agent.agent_dqna
import libs.libs_agent.agent_dqn_curiosity
import libs.libs_agent.agent_reinforce

import libs.libs_agent.agent_actor_critic_state_value

import libs.libs_rysy_python.rysy as rysy

import numpy


class AtariRLDqn:

    def __init__(self, env, network_path, agent_type = "dqn", ):
        self.env = env
        self.network_path = network_path


        #init DQN agent
        gamma = 0.99
        replay_buffer_size  = 8192
        epsilon_start       = 1.0
        epsilon_end         = 0.1
        epsilon_decay       = 0.99999
        curiosity_ratio     = 0.1

        self.training_games_to_play = 0

        if agent_type == "dqn":
            self.agent = libs.libs_agent.agent_dqn.DQNAgent(self.env, self.network_path + "network_config.json", gamma, replay_buffer_size, epsilon_start, epsilon_end, epsilon_decay)
        elif agent_type == "dqn_dueling":
            self.agent = libs.libs_agent.agent_dqn_dueling.DQNDuelingAgent(self.env, self.network_path + "network_config.json", gamma, replay_buffer_size, epsilon_start, epsilon_end, epsilon_decay)
        elif agent_type == "dqrn":
            self.agent = libs.libs_agent.agent_dqn.DQNAgent(self.env, self.network_path + "network_config.json", gamma, replay_buffer_size, epsilon_start, epsilon_end, epsilon_decay, True)
        elif agent_type == "dqna":
            self.agent = libs.libs_agent.agent_dqna.DQNAAgent(env, self.network_path, gamma, replay_buffer_size, epsilon_start, epsilon_end, epsilon_decay)
        elif agent_type == "curiosity":
            self.agent = libs.libs_agent.agent_dqn_curiosity.DQNCuriosityAgent(env, self.network_path, gamma, curiosity_ratio, epsilon_start, epsilon_end, epsilon_decay)
        elif agent_type == "reinforce":
            self.agent = libs.libs_agent.agent_reinforce.Reinforce(env, self.network_path, gamma, replay_buffer_size, epsilon_start, epsilon_end, epsilon_decay)
        elif agent_type == "actor_critic_state_value":
            self.agent = libs.libs_agent.agent_actor_critic_state_value.ActorCriticStateValue(env, self.network_path, gamma, replay_buffer_size, epsilon_start, epsilon_end, epsilon_decay)



    def visualise(self, put_heatmap = False):
        self.agent.load(self.network_path)

        self.agent.run_best_enable()

        while True:
            self.agent.main()

            if put_heatmap:
                heat_map = self.agent.get_heatmap()
                self.env.set_heat_map(heat_map)
            self.env.render()
            self.env._print()

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
                self.agent.save(self.network_path)

        self.agent.save(self.network_path)


    def test(self, log_filename_prefix, testing_games_to_play = 100):
        self.agent.load(self.network_path)

        #choose only the best action
        self.agent.run_best_enable()

        score = []
        game_id    = 0

        #process testing games
        while self.env.get_games_count() < testing_games_to_play + self.training_games_to_play:
            self.agent.main()

            if self.env.get_games_count() != game_id:
                game_id = self.env.get_games_count()
                score.append(self.env.get_score())

                self.env.reset_score()

                print(score)


        mean_score = numpy.mean(score)
        std        = numpy.std(score)

        result = "games count : " + str(len(score)) + "\n"
        result+= "mean score : " + str(mean_score) + "\n"
        result+= "std score : " + str(std) + "\n"


        result+= "games : " + "\n"
        for i in range(0, len(score)):
            result+= str(score[i]) + "\n"

        testing_progress_log = rysy.Log(self.network_path + log_filename_prefix + "result_testing.log")
        testing_progress_log.put_string(result)
