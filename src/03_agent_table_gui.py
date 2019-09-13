#table agent example with opengl GUI
#learn avoid the cliff
#try to compare Q-learning and SARSA trajectory results

import libs.libs_env.env_cliff_gui
import libs.libs_agent.agent_table

#init cliff environment
env = libs.libs_env.env_cliff_gui.EnvCliffGui()

#print environment info
env.print_info()

#init Q Learning agent
agent = libs.libs_agent.agent_table.QLearningAgent(env)

#init sarsa agent
#agent = libs_agent.agent_table.SarsaAgent(env)

#process training
training_iterations = 10000

for iteration in range(0, training_iterations):
    agent.main()
    #print training progress %, ane score, every 100th iterations
    if iteration%100 == 0:
        print(iteration*100.0/training_iterations, env.get_score())

#reset score
env.reset_score()

#choose only the best action
agent.run_best_enable()


#process testing iterations or infinite loop

#for iteration in range(0, 2000):
while True:
    agent.main()

    print("move=", env.get_move(), " score=",env.get_score())
    env.render()

print("program done")
print("move=", env.get_move(), " score=",env.get_score())
