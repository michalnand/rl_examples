import libs.libs_env.env_atari_enduro
import atari_rl_dqn


#init environment
env = libs.libs_env.env_atari_enduro.EnvAtariEnduro(48)


'''
network_path = "networks/atari/enduro/"
rl_atari = atari_rl_dqn.AtariRLDqn(env, network_path)
'''

network_path = "networks/atari/enduro_dueling_dqn/"
rl_atari = atari_rl_dqn.AtariRLDqn(env, network_path, "dqn_dueling")

#rl_atari.train(1000)
rl_atari.visualise()
