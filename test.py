import numpy

# 2 in, 3 hidden, 1 out. xor, because xor is the smallest thing that
# needs a hidden layer at all

X = numpy.array([[0, 0],
                 [0, 1],
                 [1, 0],
                 [1, 1]])
y = numpy.array([[0], [1], [1], [0]])

numpy.random.seed(1)
W1 = numpy.random.random((2, 3)) - 0.5
b1 = numpy.zeros((1, 3))
W2 = numpy.random.random((3, 1)) - 0.5
b2 = numpy.zeros((1, 1))

LR = 0.5

i = 0
while i < 10000:
    h = 1.0 / (1.0 + numpy.exp(-(numpy.dot(X, W1) + b1)))
    o = 1.0 / (1.0 + numpy.exp(-(numpy.dot(h, W2) + b2)))

    # o*(1-o) is the sigmoid derivative, you get it out of o itself which
    # is the whole reason everyone uses sigmoid
    e = o - y
    do = e * o * (1 - o)
    dh = numpy.dot(do, W2.T) * h * (1 - h)

    # MINUS!!! do points the way the error gets BIGGER. see notes2
    W2 = W2 - LR * numpy.dot(h.T, do)
    b2 = b2 - LR * do.sum(axis=0)
    W1 = W1 - LR * numpy.dot(X.T, dh)
    b1 = b1 - LR * dh.sum(axis=0)

    if i % 1000 == 0:
        print(i, ((o - y)**2).mean())
    i = i + 1

print(o)
