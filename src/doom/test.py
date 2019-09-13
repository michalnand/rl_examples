import env_doom
import libs.libs_agent.agent
import time

#env = env_doom.EnvDoom("basic")
#env = env_doom.EnvDoom("health_gathering")
#env = env_doom.EnvDoom("defend_the_center")
#env = env_doom.EnvDoom("defend_the_line")
env = env_doom.EnvDoom("deadly_corridor")
#env = env_doom.EnvDoom("deathmatch")
env.print_info()

agent = libs.libs_agent.agent.Agent(env)

while True:
    agent.main()

    env.render_state(0)
    #if env.get_reward() != 0:
    #    print(env.get_iterations(), "reward = ", env.get_reward(), "\n\n")

    if env.get_iterations()%256 == 0:
        env._print()
    #time.sleep(0.01)
