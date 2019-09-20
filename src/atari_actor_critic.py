import libs.libs_env.env_atari_arkanoid
import atari_rl_dqn


#init environment
env = libs.libs_env.env_atari_arkanoid.EnvAtariArkanoid(48)

network_path = "networks/atari/actor_critic_a/"
#network_path = "networks/atari/actor_critic_b/"
rl_atari = atari_rl_dqn.AtariRLDqn(env, network_path, "actor_critic")

rl_atari.train(200)
#rl_atari.visualise()
