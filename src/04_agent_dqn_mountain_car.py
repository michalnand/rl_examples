#mountain car problem
#basic problem where deep Q network can be used
#network has 2 inputs (position, velocity)
#8 hidden ELU neurons
#3 output neurons
#mountain car problem is long episodic problem
#eperience buffer should be huge (4096) and gamma = 1

import libs.libs_env.env_mountain_car
#import libs.libs_agent.agent_dqn
import libs.libs_agent.agent_dqn

#init cliff environment
env = libs.libs_env.env_mountain_car.EnvMountainCar()

#print environment info
env.print_info()


#init DQN agent


gamma = 1.0
replay_buffer_size  = 4096
epsilon_training    = 0.2
epsilon_testing     = 0.01

agent = libs.libs_agent.agent_dqn.DQNAgent(env, "networks/mountain_car/parameters.json", gamma, replay_buffer_size, epsilon_training, epsilon_testing)

#process training
training_iterations = 500000

for iteration in range(0, training_iterations):
    agent.main()
    #print training progress %, ane score, every 100th iterations
    if iteration%100 == 0:
        print(iteration*100.0/training_iterations, env.get_score())

#agent.save("networks/mountain_car/trained/")

#agent.load("networks/mountain_car/trained/")

#reset score
env.reset_score()

#choose only the best action
agent.run_best_enable()


#process testing iterations
testing_iterations = 10000
for iteration in range(0, testing_iterations):
    agent.main()
    print("move=", env.get_move(), " score=",env.get_score(), " moves to top = ", env.get_move_to_top())

while True:
    agent.main()
    env.render()

print("program done")
print("move=", env.get_move(), " score=",env.get_score(), " moves to top = ", env.get_move_to_top())
