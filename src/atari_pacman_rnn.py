import libs.libs_env.env_atari_pacman
import atari_rl_dqn


#init environment
env = libs.libs_env.env_atari_pacman.EnvAtariPacman(48)


network_path = "networks/atari/pacman_rnn/"
rl_atari = atari_rl_dqn.AtariRLDqn(env, network_path, "dqrn")

#rl_atari.train(1000)
rl_atari.visualise()
