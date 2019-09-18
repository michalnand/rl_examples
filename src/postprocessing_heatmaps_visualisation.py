import libs.libs_agent.agent
import libs.libs_agent.agent_dqn
from libs.libs_rysy_python.rysy import *

import libs.libs_env.env_atari_arkanoid
import libs.libs_env.env_atari_enduro
import libs.libs_env.env_atari_invaders
import libs.libs_env.env_atari_pacman
import libs.libs_env.env_atari_pong
import libs.libs_env.env_atari_snake




def attention_visualisation(env, network_path, output_prefix):

    state_shape  = Shape(env.get_width(), env.get_height(), env.get_depth()*env.get_time())
    output_shape = Shape(1, 1, env.get_actions_count())

    network = CNN(network_path + "network_config.json", state_shape, output_shape)
    network.load_weights(network_path + "trained/")

    state_vector = VectorFloat(env.get_observation())
    network.heatmap_visualisation(network_path + "attention_heatmap/" + output_prefix, state_vector)

def process_env(env, prefix):
    iterations = 1000
    for i in range(0, iterations):
        env.do_random_action()
        if i%(iterations/10) == 0:
            attention_visualisation(env, "networks/atari/" + prefix + "/", str(i))
            attention_visualisation(env, "networks/atari_multi/multi_network_0/", prefix + "/" + str(i))
            #attention_visualisation(env, "networks/atari_multi/multi_network_1/", prefix + "/" + str(i))
            #attention_visualisation(env, "networks/atari_multi/multi_network_2/", prefix + "/" + str(i))


env = libs.libs_env.env_atari_arkanoid.EnvAtariArkanoid(48)
process_env(env, "arkanoid")

env = libs.libs_env.env_atari_enduro.EnvAtariEnduro(48)
process_env(env, "enduro")

env = libs.libs_env.env_atari_invaders.EnvAtariInvaders(48)
process_env(env, "invaders")

env = libs.libs_env.env_atari_pacman.EnvAtariPacman(48)
process_env(env, "pacman")

env = libs.libs_env.env_atari_pong.EnvAtariPong(48)
process_env(env, "pong")

env = libs.libs_env.env_atari_snake.EnvAtariSnake(48)
process_env(env, "snake")
