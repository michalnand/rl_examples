#pong game duel
#playing two networks against each other

import libs.libs_env.env_pong_duel
import libs.libs_agent.agent_dqn

#init cliff environment
env = libs.libs_env.env_pong_duel.EnvPongDuel()

#print environment info
env.print_info()


#init DQN agent
#you can choose from pre-saved neteworks a, b, c
agent_0 = libs.libs_agent.agent_dqn.DQNAgent(env, "networks/pong_network_b/parameters.json", 0.2, 0.01, 0.99999)
agent_1 = libs.libs_agent.agent_dqn.DQNAgent(env, "networks/pong_network_b/parameters.json", 0.2, 0.01, 0.99999)

agent_0.load("networks/pong_network_b/trained/")
agent_1.load("networks/pong_network_b/trained/")

#reset score
env.reset_score()

#choose only the best action
agent_0.run_best_enable()
agent_1.run_best_enable()

while True:
    agent_0.main()
    agent_1.main()
    env.render()
