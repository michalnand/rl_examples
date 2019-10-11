import numpy


numpy.set_printoptions(formatter={'float': lambda x: "{0:0.4f}".format(x)})

def create_vector(length, range = 5.0):
    result = numpy.random.uniform(-range, range, length)
    return result

def softmax(v):
    tmp = v - numpy.max(v)
    result = numpy.exp(tmp)
    result/= numpy.sum(result)

    return result

def grad_softmax(probs):
    s = probs.reshape(-1,1)
    return numpy.diagflat(s) - numpy.dot(s, s.T)

def entropy_grad(probs):
    entropy_grad = numpy.log2(probs) + 1.0/numpy.log(2.0)
    return numpy.dot(grad_softmax(probs), -entropy_grad)

def entropy(probs):
    h = -numpy.sum((probs*numpy.log2(probs)))
    return h


count = 4
values       = create_vector(count)


for i in range(0, 1000):

    probs = softmax(values)
    if i%10 == 0:
        print(values, probs, entropy(probs))


    values+= 0.1*entropy_grad(probs)
