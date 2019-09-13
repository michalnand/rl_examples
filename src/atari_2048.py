import libs.libs_env.env_atari_2048
import atari_rl_dqn


#init environment
env = libs.libs_env.env_atari_2048.EnvAtari2048(48)


network_path = "networks/atari/2048/"
rl_atari = atari_rl_dqn.AtariRLDqn(env, network_path)

#rl_atari.train(1000)
#rl_atari.test()
rl_atari.visualise()
