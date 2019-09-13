#this is basic environment test
#create example environment and agent playing random strategy

import libs.libs_env.env_cliff
import libs.libs_agent.agent

#init cliff environment
env = libs.libs_env.env_cliff.EnvCliff()

#print environment info
env.print_info()

#init dummy agent - doing only random actions
agent = libs.libs_agent.agent.Agent(env)

#simulate training -> random moves only
training_iterations = 10000

for iteration in range(0, training_iterations):
    agent.main()
    #print training progress %, ane score, every 100th iterations
    if iteration%100 == 0:
        print(iteration*100.0/training_iterations, env.get_score())

#reset score
env.reset_score()

#choose only the best action -> doesn't matter on this agent
agent.run_best_enable()


#process testing iterations
testing_iterations = 1000
for iteration in range(0, testing_iterations):
    #process agent
    agent.main()

    #draw
    env.render()

print("program done")
print("move=", env.get_move(), " score=",env.get_score())
