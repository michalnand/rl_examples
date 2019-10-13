import libs.libs_env.env_mountain_car
import libs.libs_env.env_atari_pong
import libs.libs_env.env_atari_snake
import libs.libs_env.env_atari_arkanoid
import atari_rl_dqn


import libs.libs_env.env_pong
import libs.libs_agent.agent

#init environment
#env = libs.libs_env.env_mountain_car.EnvMountainCar()
#env = libs.libs_env.env_atari_arkanoid.EnvAtariArkanoid(24)
env = libs.libs_env.env_pong.EnvPong(9)
#env = libs.libs_env.env_atari_snake.EnvAtariSnake(24)

'''
network_path = "networks/dqn/"
rl_atari = atari_rl_dqn.AtariRLDqn(env, network_path, "dqn")

rl_atari.train(1000)
#rl_atari.visualise()
'''

'''
network_path = "networks/dqn_dueling/"
rl_atari = atari_rl_dqn.AtariRLDqn(env, network_path, "dqn_dueling")

rl_atari.train(200)
rl_atari.visualise()
'''

network_path = "networks/actor_critic/"
rl_atari = atari_rl_dqn.AtariRLDqn(env, network_path, "actor_critic")

#rl_atari.train(1000)
rl_atari.visualise()
