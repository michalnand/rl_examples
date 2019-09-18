from libs.libs_rysy_python.rysy import *



def kernel_visualisation(network_path):

    network = CNN(network_path + "trained/network_config.json")
    network.load_weights(network_path + "trained/")
    network._print()

    network.kernel_visualisation(network_path + "kernel_visualisation/");


'''
kernel_visualisation("networks/atari/arkanoid/")
kernel_visualisation("networks/atari/enduro/")
kernel_visualisation("networks/atari/invaders/")
kernel_visualisation("networks/atari/pacman/")
kernel_visualisation("networks/atari/pong/")
kernel_visualisation("networks/atari/snake/")
'''

kernel_visualisation("networks/atari_multi/multi_network_0/")
kernel_visualisation("networks/atari_multi/multi_network_1/")
kernel_visualisation("networks/atari_multi/multi_network_2/")
