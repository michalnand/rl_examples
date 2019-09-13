import libs.libs_env.env_atari_invaders
import atari_rl_dqn


#init environment
env = libs.libs_env.env_atari_invaders.EnvAtariInvaders(48)


network_path = "networks/atari/invaders/"
rl_atari = atari_rl_dqn.AtariRLDqn(env, network_path)

#rl_atari.train(1000)
#rl_atari.test()
rl_atari.visualise()

#rl_atari.kernel_visualisation()
#rl_atari.activity_visualisation()
