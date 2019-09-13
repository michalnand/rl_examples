import sys
sys.path.append("..") # Adds higher directory to python modules path.

import libs.libs_env.env_atari_arkanoid
import agent
import agent_dqn


#init environment
env = libs.libs_env.env_atari_arkanoid.EnvAtariArkanoid()

#print environment info
env.print_info()


'''
#random play environment test
random_agent = agent.Agent(env)
while True:
    random_agent.main()

    if env.get_iterations()%256 == 0:
        print("  miss ",  env.get_miss(),  " iterations = ",  env.get_iterations())
        env.render()
'''
#init DQN agent
dqn_agent = agent_dqn.DQNAgent(env, "atari_arkanoid_network.json", 0.4, 0.1, 0.99999)


#process training
total_games_to_play = 500


while env.get_games_count() < total_games_to_play:
    dqn_agent.main()

    #print training progress %, ane score, every 100th iterations
    if env.get_iterations()%256 == 0:
        env._print()
        env.render()


    if env.get_iterations()%256 == 0:
        print("done = ", env.get_games_count()*100.0/total_games_to_play, "%", "  miss ",  env.get_miss(), " eps = ", dqn_agent.get_epsilon_training(), " iterations = ",  env.get_iterations())


#reset score
env.reset_score()

#choose only the best action
dqn_agent.run_best_enable()


#process testing iterations
testing_games_to_play = 10000
while env.get_games_count() < total_games_to_play + testing_games_to_play:
    dqn_agent.main()
    env._print()


while True:
    agent.main()
    env.render()

print("program done")
print("move=", env.get_move(), " score=",env.get_score())
