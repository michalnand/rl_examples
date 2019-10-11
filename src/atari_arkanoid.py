import libs.libs_env.env_atari_arkanoid
import atari_rl_dqn


#init environment
env = libs.libs_env.env_atari_arkanoid.EnvAtariArkanoid(48)

'''
network_path = "networks/atari/arkanoid/"
rl_atari = atari_rl_dqn.AtariRLDqn(env, network_path)
'''

'''
network_path = "networks/atari/arkanoid_dueling_dqn/"
rl_atari = atari_rl_dqn.AtariRLDqn(env, network_path, "dqn_dueling")
'''



network_path = "networks/atari/arkanoid_a2c/"
rl_atari = atari_rl_dqn.AtariRLDqn(env, network_path, "actor_critic")


rl_atari.train(1000)
#rl_atari.visualise()
