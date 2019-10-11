import libs.libs_agent.agent_dqn
import libs.libs_agent.agent_dqn_dueling
import libs.libs_agent.agent_actor_critic

import libs.libs_rysy_python.rysy as rysy

import numpy


class AtariRLDqn:

    def __init__(self, env, agent_config_path, agent_type = "dqn", ):
        self.env = env
        self.agent_config_path = agent_config_path

        self.training_games_to_play = 0

        if agent_type == "dqn":
            self.agent = libs.libs_agent.agent_dqn.DQNAgent(self.env, self.agent_config_path)
        elif agent_type == "dqn_dueling":
            self.agent = libs.libs_agent.agent_dqn_dueling.DQNDuelingAgent(self.env, self.agent_config_path)
        elif agent_type == "actor_critic":
            self.agent = libs.libs_agent.agent_actor_critic.ActorCritic(env, self.agent_config_path)



    def visualise(self, put_heatmap = False):
        self.agent.load(self.agent_config_path)

        self.agent.run_best_enable()

        while True:
            self.agent.main()

            if put_heatmap:
                heat_map = self.agent.get_heatmap()
                self.env.set_heat_map(heat_map)
            self.env.render()
            self.env._print()

    def train(self, training_games_to_play = 500):
        training_progress_log = rysy.Log(self.agent_config_path + "progress_training.log")

        self.training_games_to_play = training_games_to_play
        #process training
        while self.env.get_games_count() < self.training_games_to_play:
            self.agent.main()

            #print training progress %, ane score, every 256th iterations
            if self.env.get_iterations()%256 == 0:
                str_progress = str(self.env.get_iterations()) + " "
                str_progress+= str(self.env.get_games_count()) + " "
                str_progress+= str(self.env.get_score()) + " "
                str_progress+= "\n"
                training_progress_log.put_string(str_progress)

                print("done = ", self.env.get_games_count()*100.0/self.training_games_to_play, "%", " iterations = ",  self.env.get_iterations(), " score = ",  self.env.get_score())

            if self.env.get_iterations()%50000 == 0:
                print("SAVING network")
                self.agent.save(self.agent_config_path)

        self.agent.save(self.agent_config_path)


    def test(self, log_filename_prefix, testing_games_to_play = 100):
        self.agent.load(self.agent_config_path)

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

        testing_progress_log = rysy.Log(self.agent_config_path + log_filename_prefix + "result_testing.log")
        testing_progress_log.put_string(result)
