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

h = 1.0 / (1.0 + numpy.exp(-(numpy.dot(X, W1) + b1)))
o = 1.0 / (1.0 + numpy.exp(-(numpy.dot(h, W2) + b2)))

print h
print o
print ((o - y)**2).mean()
