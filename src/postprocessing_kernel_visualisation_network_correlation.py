import json
import numpy

epsilon = 0.0000000001

def load_json(json_file_name):
    json_file = open(json_file_name)
    data = json.load(json_file)

    width  = int(data["shape"][0])
    height = int(data["shape"][1])
    depth  = int(data["shape"][2])

    kernels_count = int(data["kernels_count"])

    result = numpy.zeros((kernels_count, depth, height, width))

    for kernel in range(0, kernels_count):
        for d in range(0, depth):
            for h in range(0, height):
                for w in range(0, width):
                    value = float(data["result"][kernel][d][h][w])
                    result[kernel][d][h][w] = value


    return result


def kernel_correlation(kernel_a, kernel_b):
    tmp_a = kernel_a.copy()
    tmp_b = kernel_b.copy()

    tmp_a = numpy.reshape(tmp_a, tmp_a.size)
    tmp_b = numpy.reshape(tmp_b, tmp_b.size)


    tmp_a-= tmp_a.mean();
    tmp_b-= tmp_b.mean();
    tmp_a/= numpy.linalg.norm(tmp_a) + epsilon
    tmp_b/= numpy.linalg.norm(tmp_b) + epsilon

    res = numpy.dot(tmp_a, tmp_b)

    return res



def layer_correlation(layer_a, layer_b):
    kernels_count = len(layer_a)
    corr_mat = numpy.zeros((kernels_count, kernels_count))

    for a in range(0, kernels_count):
        for b in range(0, kernels_count):
            corr_mat[a][b] = kernel_correlation(layer_a[a], layer_b[b])


    result = 0.0
    for a in range(0, kernels_count):
        result+= corr_mat[a].max()
    result/= kernels_count

    return result


def network_correlation(network_a, network_b, layers_inidicies = [0, 3, 6, 9]):

    result = 0.0

    for i in range(0, len(layers_inidicies)):
        layer_name_a = network_a + str(layers_inidicies[i]) + ".json"
        layer_name_b = network_b + str(layers_inidicies[i]) + ".json"

        layer_a = load_json(layer_name_a)
        layer_b = load_json(layer_name_b)

        result+= layer_correlation(layer_a, layer_b)

    return result



networks = []

networks.append("networks/atari/arkanoid/kernel_visualisation/")
networks.append("networks/atari/enduro/kernel_visualisation/")
networks.append("networks/atari/invaders/kernel_visualisation/")
networks.append("networks/atari/pacman/kernel_visualisation/")
networks.append("networks/atari/pong/kernel_visualisation/")
networks.append("networks/atari/snake/kernel_visualisation/")

networks.append("networks/atari_multi/multi_network_0/kernel_visualisation/")


networks_count = len(networks)
result = numpy.zeros((networks_count, networks_count))

for a in range(0, networks_count):
    print("processing ", networks[a])
    for b in range(0, networks_count):
        print("   with ", networks[b])

        net_corr = network_correlation(networks[a], networks[b])
        result[a][b] = net_corr
    result[a]/= result[a].max() + epsilon


total_corr = numpy.zeros((networks_count))
for a in range(0, networks_count):
    for b in range(0, networks_count):
        total_corr[a]+= result[a][b]

print("\n\n")
print("networks correlations")
for a in range(0, networks_count):
    print(networks[a], " : ", end = " ")
    for b in range(0, networks_count):
        print(round(result[a][b], 4), end = " ")
    print()

print("\n\n")
print("total correlation : ")
for i in range(0, networks_count):
    print(networks[i], round(total_corr[i], 4))
