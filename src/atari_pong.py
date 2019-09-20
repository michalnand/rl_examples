import libs.libs_env.env_atari_pong
import atari_rl_dqn


#init environment
env = libs.libs_env.env_atari_pong.EnvAtariPong(48)


network_path = "networks/atari/pong/"
rl_atari = atari_rl_dqn.AtariRLDqn(env, network_path)

#rl_atari.train(1000)
rl_atari.visualise()
