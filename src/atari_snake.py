import libs.libs_env.env_atari_snake
import atari_rl_dqn


#init environment
env = libs.libs_env.env_atari_snake.EnvAtariSnake(48)


network_path = "networks/atari/snake/"
rl_atari = atari_rl_dqn.AtariRLDqn(env, network_path)

#rl_atari.train(1000)
#rl_atari.test()
rl_atari.visualise()

#rl_atari.kernel_visualisation()
#rl_atari.activity_visualisation()
