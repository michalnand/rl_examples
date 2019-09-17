import libs.libs_env.env_atari_arkanoid
import libs.libs_env.env_atari_enduro
import libs.libs_env.env_atari_invaders
import libs.libs_env.env_atari_pacman
import libs.libs_env.env_atari_pong
import libs.libs_env.env_atari_snake

import atari_rl_dqn



def test(env_id, network_path):
    size = 48

    if env_id == 0:
        env = libs.libs_env.env_atari_arkanoid.EnvAtariArkanoid(size)
    elif env_id == 1:
        env = libs.libs_env.env_atari_enduro.EnvAtariEnduro(size)
    elif env_id == 2:
        env = libs.libs_env.env_atari_invaders.EnvAtariInvaders(size)
    elif env_id == 3:
        env = libs.libs_env.env_atari_pacman.EnvAtariPacman(size)
    elif env_id == 4:
        env = libs.libs_env.env_atari_pong.EnvAtariPong(size)
    elif env_id == 5:
        env = libs.libs_env.env_atari_snake.EnvAtariSnake(size)

    rl_atari = atari_rl_dqn.AtariRLDqn(env, network_path)
    rl_atari.test(str(env_id) + "_")


test(0, "networks/atari/arkanoid/")
test(1, "networks/atari/enduro/")
test(2, "networks/atari/invaders/")
test(3, "networks/atari/pacman/")
test(4, "networks/atari/pong/")
test(5, "networks/atari/snake/")

test(0, "networks/atari_multi/multi_network_0/")
test(1, "networks/atari_multi/multi_network_0/")
test(2, "networks/atari_multi/multi_network_0/")
test(3, "networks/atari_multi/multi_network_0/")
test(4, "networks/atari_multi/multi_network_0/")
test(5, "networks/atari_multi/multi_network_0/")

test(0, "networks/atari_multi/multi_network_1/")
test(1, "networks/atari_multi/multi_network_1/")
test(2, "networks/atari_multi/multi_network_1/")
test(3, "networks/atari_multi/multi_network_1/")
test(4, "networks/atari_multi/multi_network_1/")
test(5, "networks/atari_multi/multi_network_1/")

test(0, "networks/atari_multi/multi_network_2/")
test(1, "networks/atari_multi/multi_network_2/")
test(2, "networks/atari_multi/multi_network_2/")
test(3, "networks/atari_multi/multi_network_2/")
test(4, "networks/atari_multi/multi_network_2/")
test(5, "networks/atari_multi/multi_network_2/")
