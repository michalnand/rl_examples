import sys
sys.path.append("..") # Adds higher directory to python modules path.

import libs.libs_env.blackbox.env_black_box as env_black_box
import libs.libs_agent.agent_dqn



#init environment
env = env_black_box.EnvBlackBox(4)

#print environment info
env.print_info()


gamma = 0.95
replay_buffer_size  = 256
epsilon_training    = 1.0
epsilon_testing     = 0.05
epsilon_decay       = 0.9999

agent = libs.libs_agent.agent_dqn.DQNAgent(env, "networks/black_box_network/network_config.json", gamma, replay_buffer_size, epsilon_training, epsilon_testing, epsilon_decay)



#process training
training_iterations = 100000


for i in range(0, training_iterations):
    agent.main()

    if env.get_iterations()%256 == 0:
        print(" iterations = ",  env.get_iterations(), " score = ", env.get_score(), " epsilon = ", agent.get_epsilon_training())


#reset score
env.reset_score()

#choose only the best action
agent.run_best_enable()


#process testing iterations
testing_iterations = 10000
for i in range(0, testing_iterations):
    agent.main()

print("move=", env.get_move(), " score=",env.get_score())
print("program done")
