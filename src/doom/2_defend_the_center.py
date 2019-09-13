import env_doom
import libs.libs_agent.agent_dqn
import libs.libs_agent.agent
import libs.libs_rysy_python.rysy as rysy

network_path = "network_defend_the_center/"
env = env_doom.EnvDoom("defend_the_center")
env.print_info()

#init DQN agent
gamma = 0.99
replay_buffer_size  = 16384
epsilon_training    = 1.0
epsilon_testing     = 0.1
epsilon_decay       = 0.99999

#init DQN agent
agent = libs.libs_agent.agent_dqn.DQNAgent(env, network_path + "network_config.json", gamma, replay_buffer_size, epsilon_training, epsilon_testing, epsilon_decay)

training_progress_log = rysy.Log(network_path + "progress_training.log")
testing_progress_log = rysy.Log(network_path + "progress_testing.log")

#process training
total_games_to_play = 12000
while env.get_games_count() < total_games_to_play:
    agent.main()


    if env.get_iterations()%256 == 0:
        str_progress = str(env.get_iterations()) + " "
        str_progress+= str(env.get_games_count()) + " "
        str_progress+= str(agent.get_epsilon_training()) + " "
        str_progress+= str(env.get_score()) + " "
        str_progress+= str(env.get_kill_count()) + " "
        str_progress+= str(env.get_death_count()) + " "
        str_progress+= str(env.get_game_kd_ratio()) + " "
        str_progress+= str(env.get_kd_ratio()) + " "
        str_progress+= "\n"
        training_progress_log.put_string(str_progress)

        print("done = ", env.get_games_count()*100.0/total_games_to_play, "%", " eps = ", agent.get_epsilon_training(), " iterations = ",  env.get_iterations())

    if env.get_iterations()%50000 == 0:
        print("SAVING network")
        agent.save(network_path + "trained/")

agent.save(network_path + "trained/")


agent.load(network_path + "trained/")



#reset score
env.reset_score()
env.reset()

#choose only the best action
agent.run_best_enable()


#process testing games
testing_games_to_play = 500
while env.get_games_count() < total_games_to_play + testing_games_to_play:
    agent.main()

    if env.get_iterations()%256 == 0:
        str_progress = str(env.get_iterations()) + " "
        str_progress+= str(env.get_games_count() - total_games_to_play) + " "
        str_progress+= str(agent.get_epsilon_training()) + " "
        str_progress+= str(env.get_score()) + " "
        str_progress+= str(env.get_kill_count()) + " "
        str_progress+= str(env.get_death_count()) + " "
        str_progress+= str(env.get_game_kd_ratio()) + " "
        str_progress+= str(env.get_kd_ratio()) + " "
        str_progress+= "\n"
        testing_progress_log.put_string(str_progress)

print("TESTING SCORE =", env.get_score())
print("program done")
