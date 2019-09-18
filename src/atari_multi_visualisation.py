import libs.libs_env.env_atari_multi
import atari_multi_rl_dqn
import atari_rl_dqn

import libs.libs_env.env_atari_arkanoid
import libs.libs_env.env_atari_enduro
import libs.libs_env.env_atari_invaders
import libs.libs_env.env_atari_pacman
import libs.libs_env.env_atari_pong
import libs.libs_env.env_atari_snake


def run(network_path, agent_type):
    envs = []
    size = 48


    envs.append(libs.libs_env.env_atari_arkanoid.EnvAtariArkanoid(size))
    envs.append(libs.libs_env.env_atari_enduro.EnvAtariEnduro(size))
    envs.append(libs.libs_env.env_atari_invaders.EnvAtariInvaders(size))
    envs.append(libs.libs_env.env_atari_pacman.EnvAtariPacman(size))
    envs.append(libs.libs_env.env_atari_pong.EnvAtariPong(size))
    envs.append(libs.libs_env.env_atari_snake.EnvAtariSnake(size))

    #init environment
    env = libs.libs_env.env_atari_multi.EnvAtariMulti(envs, size, 512)

    rl_atari = atari_rl_dqn.AtariRLDqn(env, network_path, agent_type)
    rl_atari.visualise()
 

#run("networks/atari_multi/multi_network_0/", "dqn")
#run("networks/atari_multi/multi_network_1/", "dqn")
run("networks/atari_multi/multi_network_2/", "dqn")
