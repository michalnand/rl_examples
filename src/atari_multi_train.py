import libs.libs_env.env_atari_multi
import atari_multi_rl_dqn

import libs.libs_env.env_atari_arkanoid
import libs.libs_env.env_atari_enduro
import libs.libs_env.env_atari_invaders
import libs.libs_env.env_atari_pacman
import libs.libs_env.env_atari_pong
import libs.libs_env.env_atari_snake


def run_experiment(network_path, training_games, agent_type):
    envs = []
    size = 48

    envs.append(libs.libs_env.env_atari_arkanoid.EnvAtariArkanoid(size))
    envs.append(libs.libs_env.env_atari_enduro.EnvAtariEnduro(size))
    envs.append(libs.libs_env.env_atari_invaders.EnvAtariInvaders(size))
    envs.append(libs.libs_env.env_atari_pacman.EnvAtariPacman(size))
    envs.append(libs.libs_env.env_atari_pong.EnvAtariPong(size))
    envs.append(libs.libs_env.env_atari_snake.EnvAtariSnake(size))


    #init environment
    env = libs.libs_env.env_atari_multi.EnvAtariMulti(envs, size, 8192)

    rl_atari = atari_multi_rl_dqn.AtariMultiRLDqn(env, network_path, agent_type)

    rl_atari.train(training_games)
    rl_atari.test()
    #rl_atari.visualise()


#run_experiment("networks/atari_multi/multi_network_0/", 1000, "dqn")
run_experiment("networks/atari_multi/multi_network_1/", 1000, "dqn")
#run_experiment("networks/atari_multi/multi_network_2/", 1000, "dqn")
