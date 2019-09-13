#pong game
#game is played against hard coded still winning bot
#reward is -1 when miss, and +1 when hit the ball
#example for convolutional and deep neural network use

import libs.libs_env.env_gym
import libs.libs_agent.agent
import libs.libs_agent.agent_dqn
import libs.libs_rysy_python.rysy as rysy

network_path = "networks/atari_network_0/"
#network_path = "networks/gym_breakout_network/"
#network_path = "networks/gym_space_invaders_network/"

#init cliff environment
env = libs.libs_env.env_gym.EnvGym("Pong-v0")
#env = libs.libs_env.env_gym.EnvGym("Breakout-v0")
#env = libs.libs_env.env_gym.EnvGym("SpaceInvaders-v0")

#print environment info
env.print_info()

progress_training_log = rysy.Log(network_path + "progress_training.log")
progress_testing_log = rysy.Log(network_path + "progress_testing.log")

render_enabled = False


#DQN agent parameters
gamma = 0.99
replay_buffer_size  = 8192
epsilon_training    = 1.0
epsilon_testing     = 0.1
epsilon_decay       = 0.99999

#init DQN agent
agent = libs.libs_agent.agent_dqn.DQNAgent(env, network_path + "network_config.json", gamma, replay_buffer_size, epsilon_training, epsilon_testing, epsilon_decay)



#process training
training_games = 500

iteration = 0
while env.get_game_number() < training_games:
    agent.main()
    #print training progress %, ane score, every 100th iterations
    if iteration%256 == 0:

        str_progress = str(iteration) + " "
        str_progress+= str(env.get_game_number()) + " "
        str_progress+= str(agent.get_epsilon_training()) + " "
        str_progress+= str(env.get_score()) + " "
        str_progress+= "\n"
        progress_training_log.put_string(str_progress)

        if render_enabled:
            env.render()

        print(iteration, env.get_game_number(),  env.get_game_number()*100.0/training_games, env.get_score(), agent.get_epsilon_training())


    if iteration%10000 == 0:
        print("saving")
        agent.save( network_path + "trained/")

    iteration+= 1


agent.save( network_path + "trained/")
agent.load(  network_path + "trained/")


#reset score
env.reset_score()

#choose only the best action
agent.run_best_enable()


#process testing iterations
testing_iterations = 10000
for iteration in range(0, testing_iterations):
    agent.main()
    print("move=", env.get_move(), " score=",env.get_score())

    if iteration%256 == 0:

        str_progress = str(iteration) + " "
        str_progress+= str(env.get_game_number()) + " "
        str_progress+= str(agent.get_epsilon_training()) + " "
        str_progress+= str(env.get_score()) + " "
        str_progress+= "\n"

        progress_testing_log.put_string(str_progress)

if render_enabled:
    while True:
        agent.main()
        env.render()

print("program done")
print("move=", env.get_move(), " score=",env.get_score())
