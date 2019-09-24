import libs.libs_env.env_mountain_car
import libs.libs_env.env_atari_snake
import atari_rl_dqn


#init environment
#env = libs.libs_env.env_mountain_car.EnvMountainCar()
env = libs.libs_env.env_atari_snake.EnvAtariSnake(24)

'''
network_path = "networks/dqn/"
rl_atari = atari_rl_dqn.AtariRLDqn(env, network_path, "dqn")

#rl_atari.train(100)
rl_atari.visualise()
'''

'''
network_path = "networks/reinforce/"
rl_atari = atari_rl_dqn.AtariRLDqn(env, network_path, "reinforce")

#rl_atari.train(200)
rl_atari.visualise()
'''

network_path = "networks/actor_critic/"
rl_atari = atari_rl_dqn.AtariRLDqn(env, network_path, "actor_critic")

#rl_atari.train(200)
rl_atari.visualise()
