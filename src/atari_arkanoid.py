import libs.libs_env.env_atari_arkanoid
import atari_rl_dqn


#init environment
env = libs.libs_env.env_atari_arkanoid.EnvAtariArkanoid(48)


network_path = "networks/atari/arkanoid_test/"
rl_atari = atari_rl_dqn.AtariRLDqn(env, network_path)

rl_atari.train(1000)
#rl_atari.visualise()
