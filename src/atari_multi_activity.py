import libs.libs_env.env_atari_multi
import atari_multi_rl_dqn
import atari_rl_dqn

import libs.libs_env.env_atari_arkanoid
import libs.libs_env.env_atari_enduro
import libs.libs_env.env_atari_invaders
import libs.libs_env.env_atari_pacman
import libs.libs_env.env_atari_pong
import libs.libs_env.env_atari_snake


def test_activity(network_path, env_name, agent_type = "dqn"):

    size = 48
    if env_name == "arkanoid":
        env = libs.libs_env.env_atari_arkanoid.EnvAtariArkanoid(size)
    elif env_name == "enduro":
        env = libs.libs_env.env_atari_enduro.EnvAtariEnduro(size)
    elif env_name == "invaders":
        env = libs.libs_env.env_atari_invaders.EnvAtariInvaders(size)
    elif env_name == "pacman":
        env = libs.libs_env.env_atari_pacman.EnvAtariPacman(size)
    elif env_name == "pong":
        env = libs.libs_env.env_atari_pong.EnvAtariPong(size)
    elif env_name == "snake":
        env = libs.libs_env.env_atari_snake.EnvAtariSnake(size)
    else:
        print("ERROR : unknown env : ", env_name)

    rl_atari = atari_rl_dqn.AtariRLDqn(env, network_path, agent_type)
    rl_atari.activity_visualisation(env_name + "/")

def kernel_visualisation(network_path, agent_type = "dqn"):

    envs = []
    size = 48

    envs.append(libs.libs_env.env_atari_arkanoid.EnvAtariArkanoid(size))
    envs.append(libs.libs_env.env_atari_enduro.EnvAtariEnduro(size))
    envs.append(libs.libs_env.env_atari_invaders.EnvAtariInvaders(size))
    envs.append(libs.libs_env.env_atari_pacman.EnvAtariPacman(size))
    envs.append(libs.libs_env.env_atari_pong.EnvAtariPong(size))
    envs.append(libs.libs_env.env_atari_snake.EnvAtariSnake(size))


    #init environment
    env = libs.libs_env.env_atari_multi.EnvAtariMulti(envs, size, 1024)

    rl_atari = atari_rl_dqn.AtariRLDqn(env, network_path, agent_type)
    rl_atari.kernel_visualisation()

'''
test_activity("networks/atari_multi/multi_network_0/", "arkanoid")
test_activity("networks/atari_multi/multi_network_0/", "enduro")
test_activity("networks/atari_multi/multi_network_0/", "invaders")
test_activity("networks/atari_multi/multi_network_0/", "pacman")
test_activity("networks/atari_multi/multi_network_0/", "pong")
test_activity("networks/atari_multi/multi_network_0/", "snake")
'''

kernel_visualisation("networks/atari_multi/multi_network_2/")
