import sys
sys.path.append("..") # Adds higher directory to python modules path.

import libs.libs_env.blackbox.env_black_box as env_black_box
import agent
import agent_dqn


#init environment
env = env_black_box.EnvBlackBox(4)

#print environment info
env.print_info()


#random play environment test
random_agent = agent.Agent(env)
while True:
    random_agent.main()

    if env.get_iterations()%256 == 0:
        print(" iterations = ",  env.get_iterations(), " score = ", env.get_score())



#init DQN agent
dqn_agent = agent_dqn.DQNAgent(env, "black_box_network.json", 0.1, 0.05, 0.99999)


#process training
training_iterations = 100000


for i in range(0, training_iterations):
    dqn_agent.main()

    if env.get_iterations()%256 == 0:
        print(" iterations = ",  env.get_iterations(), " score = ", env.get_score(), " epsilon = ", dqn_agent.get_epsilon_training())


#reset score
env.reset_score()

#choose only the best action
dqn_agent.run_best_enable()


#process testing iterations
testing_iterations = 10000
for i in range(0, testing_iterations):
    dqn_agent.main()
    #env._print()

print("move=", env.get_move(), " score=",env.get_score())
print("program done")
