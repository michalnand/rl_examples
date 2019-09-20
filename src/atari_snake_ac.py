import libs.libs_env.env_atari_snake
import atari_rl_dqn


#init environment
env = libs.libs_env.env_atari_snake.EnvAtariSnake(24)

network_path = "networks/atari/actor_critic/"
rl_atari = atari_rl_dqn.AtariRLDqn(env, network_path, "actor_critic")

rl_atari.train(200)
rl_atari.visualise()
