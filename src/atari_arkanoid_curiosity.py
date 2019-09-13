import libs.libs_env.env_atari_arkanoid
import atari_rl_dqn_curiosity


#init environment
env = libs.libs_env.env_atari_arkanoid.EnvAtariArkanoid(48)


network_path = "networks/atari/arkanoid_curiosity_0.1/"


rl_atari = atari_rl_dqn_curiosity.AtariRLDqnCuriosity(env, network_path)

rl_atari.train(1000)
rl_atari.test()
#rl_atari.visualise()

print("program done")
