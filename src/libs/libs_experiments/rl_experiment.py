import numpy
import libs.libs_env.env_arkanoid
import libs.libs_agent.agent_dqn
from scipy import stats

class RLExperiment:

    def __init__(self, env_name, training_games_count, testing_games_count, trials_count, dqn_config_file, log_file_name_prefix):
        self.env_name               = env_name
        self.training_games_count   = training_games_count
        self.testing_games_count    = testing_games_count
        self.trials_count           = trials_count
        self.dqn_config_file        = dqn_config_file

        self.log_file_name_prefix   = log_file_name_prefix

        self.training_score         = []
        self.training_score_average = 0.0
        self.training_score_std     = 0.0
        self.training_normal_test   = 1.0

        self.testing_score         = []
        self.testing_score_average = 0.0
        self.testing_score_std     = 0.0
        self.testing_normal_test   = 1.0




    def process(self):

        total_training_games_count = 0
        total_testing_games_count  = 0

        for trial in range(0, self.trials_count):
            if self.env_name == "ARKANOID":
                env = libs.libs_env.env_arkanoid.EnvArkanoid()

            agent = libs.libs_agent.agent_dqn.DQNAgent(env, self.dqn_config_file, 0.3, 0.1, 0.99999)

            score_prev  = 0
            score_now   = 0

            games_count = env.get_games_count()

            #process training

            while env.get_games_count() < self.training_games_count:
                agent.main()

                if env.get_games_count() != games_count:
                    games_count = env.get_games_count()
                    score_prev = score_now
                    score_now = env.get_score()

                    self.training_score.append(score_now - score_prev)

                    total_training_games_count+= 1

                    if total_training_games_count >= 8:
                        self.__compute_training_score_stats()
                        self.add_to_log(self.log_file_name_prefix + "_training", trial, games_count, self.training_score_average, self.training_score_std, self.training_normal_test, self.training_score)


            #reset score
            env.reset_score()

            #choose only the best action
            agent.run_best_enable()

            score_prev  = 0
            score_now   = 0

            games_count = env.get_games_count()

            #process testing


            while env.get_games_count() < self.testing_games_count + self.training_games_count:
                agent.main()

                if env.get_games_count() != games_count:
                    games_count = env.get_games_count()
                    score_prev = score_now
                    score_now = env.get_score()

                    self.testing_score.append(score_now - score_prev)

                    total_testing_games_count+= 1

                    if total_testing_games_count >= 8:
                        self.__compute_testing_score_stats()
                        self.add_to_log(self.log_file_name_prefix + "_testing", trial, games_count, self.testing_score_average, self.testing_score_std, self.testing_normal_test, self.testing_score)


    def __compute_training_score_stats(self):
        self.training_score_average = numpy.average(self.training_score)
        self.training_score_std = numpy.std(self.training_score)
        k2, p = stats.normaltest(self.training_score)
        self.training_normal_test = p


    def __compute_testing_score_stats(self):
        self.testing_score_average = numpy.average(self.testing_score)
        self.testing_score_std = numpy.std(self.testing_score)
        k2, p = stats.normaltest(self.testing_score)
        self.testing_normal_test = p

    def add_to_log(self, log_file_name_prefix, trial, game, score_average, std, normal_test, games_score):

        full_log_file_name = log_file_name_prefix + "_full.log"
        with open(full_log_file_name, "a") as file:

            s = ""
            s+= "trial = " + str(trial) + "\n"
            s+= "game  = " + str(game)  + "\n"
            s+= "score = " + str(score_average) + "\n"
            s+= "std   = " + str(std) + "\n"
            s+= "test  = " + str(normal_test) + "\n"
            s+= "\n\n"
            s+= "games_scores = " + "\n"

            score = ""
            for i in range(0, len(games_score)):
                score+= str(games_score[i]) + " "
            score+= "\n"

            s = s + score + "\n\n\n\n"

            file.write(s)

        progress_log_file_name = log_file_name_prefix + "_progress_trial_" + str(trial) + ".log"
        with open(progress_log_file_name, "a") as file:

            s = ""
            s+= str(game) + " "
            s+= str(score_average) + " "
            s+= str(std) + " "
            s+= str(normal_test) + " "
            s+= "\n"

            file.write(s)

        games_score_histogram_log_file_name = log_file_name_prefix + "_games_score_histogram.log"
        with open(games_score_histogram_log_file_name, "w") as file:

            hist, bin_edges = numpy.histogram(games_score, density=False, bins = 32)

            s = ""
            for i in range(0, len(bin_edges) - 1):
                s+= str(bin_edges[i]) + " " + str(hist[i]) + "\n"

            file.write(s)

        summary_log_file_name = log_file_name_prefix + "_summary.log"
        with open(summary_log_file_name, "w") as file:
            s = "summary results :\n\n"

            s+= "trial = " + str(trial) + "\n"
            s+= "game  = " + str(game)  + "\n"
            s+= "score = " + str(score_average) + "\n"
            s+= "std   = " + str(std) + "\n"
            s+= "test  = " + str(normal_test) + "\n"
            s+= "\n\n"
            s+= "games_scores = " + "\n"

            score = ""
            for i in range(0, len(games_score)):
                score+= str(games_score[i]) + " "
            score+= "\n"

            s = s + score

            file.write(s)
            print(s)
