import env_doom
import libs.libs_agent.agent_dqn
import libs.libs_rysy_python.rysy as rysy

#network_path = "network_basic/"
#env = env_doom.EnvDoom("basic")
network_path = "network_defend_the_line/"
env = env_doom.EnvDoom("defend_the_line")
#network_path = "network_deadly_corridor/"
#env = env_doom.EnvDoom("deadly_corridor")

env.print_info()

#init DQN agent
gamma = 0.99
replay_buffer_size  = 2048
epsilon_training    = 1.0
epsilon_testing     = 0.1
epsilon_decay       = 0.99999

#init DQN agent
agent = libs.libs_agent.agent_dqn.DQNAgent(env, network_path)

agent.load(network_path)

#agent.run_best_enable()
#agent.kernel_visualisation(network_path + "kernel_visualisation/")
#agent.activity_visualisation(network_path + "activity_visualisation/")


#reset score
env.reset_score()
env.reset()

#choose only the best action
agent.run_best_enable()

while True:
    #env.render_state()

    res = agent.main()

    #env.render_state()
    if env.get_iterations()%256 == 0:
        env._print()
