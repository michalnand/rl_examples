import json
import numpy

def load_json(file_name):
    f = open(file_name)
    result = json.load(f)

    return result


def compute_max_activity(json):

    width  = json["shape"][0]
    height = json["shape"][1]
    depth  = json["shape"][2]

    max_activity = numpy.zeros((width, height))

    for y in range(0, height):
        for x in range(0, width):
            v_max = json["result"][0][y][x]
            id_max = 0
            for d in range(0, depth):
                v = json["result"][d][y][x]
                if v > v_max:
                    v_max = v
                    id_max = d

            max_activity[y][x] = id_max

    return max_activity


def compute_histogram(bins, values, normalise = True):
    tmp = values.flatten()
    histogram = numpy.zeros(bins)

    for i in range(0, len(tmp)):
        idx = int(tmp[i])
        histogram[idx]+= 1

    if normalise:
        sum = numpy.sum(histogram)
        histogram = histogram/sum

    sum = numpy.sum(histogram)

    return histogram

def compute_entropy(values):
    s = 0.0
    for i in range(0, len(values)):
        if values[i] > 0.0:
            s+= values[i]*numpy.log2(values[i])

    return -s

def compute(folders):

    total_entropy = 0.0
    count = 0.0
    for folder in folders:
        print(folder, end = " ")
        average_entropy = 0.0
        for layer in layers:
            json_file_name = str(folder) + str(layer) + ".json"
            json_result = load_json(json_file_name)


            max_activity = compute_max_activity(json_result)
            histogram = compute_histogram(int(json_result["shape"][2]), max_activity)
            entropy = compute_entropy(histogram)

            average_entropy+= entropy
            total_entropy+= entropy

            print("%.3f" % entropy, end = " ")

        average_entropy = average_entropy/len(layers)
        print(" S = %.3f" % average_entropy)

    total_entropy = total_entropy/(len(layers)*len(folders))
    print(" S_total = %.3f" % total_entropy)


folders_single = ["../../atari/arkanoid/activity/", "../../atari/enduro/activity/", "../../atari/invaders/activity/", "../../atari/pacman/activity/", "../../atari/pong/activity/", "../../atari/snake/activity/"]
folders_multi_0 = ["../multi_network_0/activity/arkanoid/", "../multi_network_0/activity/enduro/", "../multi_network_0/activity/invaders/", "../multi_network_0/activity/pacman/", "../multi_network_0/activity/pong/", "../multi_network_0/activity/snake/"]
folders_multi_1 = ["../multi_network_1/activity/arkanoid/", "../multi_network_1/activity/enduro/", "../multi_network_1/activity/invaders/", "../multi_network_1/activity/pacman/", "../multi_network_1/activity/pong/", "../multi_network_1/activity/snake/"]

layers = [2, 5, 8, 11]

compute(folders_single)
compute(folders_multi_0)
#compute(folders_multi_1)
