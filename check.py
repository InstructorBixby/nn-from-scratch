import numpy

# my derivative against the actual definition of a derivative
#   dE/dw = (E(w+eps) - E(w-eps)) / (2*eps)
# that cant be wrong so if they dont match, i am wrong

EPS = 0.00001

X = numpy.array([[0, 0], [0, 1], [1, 0], [1, 1]])
y = numpy.array([[0], [1], [1], [0]])

numpy.random.seed(3)
W1 = numpy.random.random((2, 3)) - 0.5
b1 = numpy.zeros((1, 3))
W2 = numpy.random.random((3, 1)) - 0.5
b2 = numpy.zeros((1, 1))

# my version, same as test.py
h = 1.0 / (1.0 + numpy.exp(-(numpy.dot(X, W1) + b1)))
o = 1.0 / (1.0 + numpy.exp(-(numpy.dot(h, W2) + b2)))
do = (o - y) * o * (1 - o)
dh = numpy.dot(do, W2.T) * h * (1 - h)
g2 = numpy.dot(h.T, do)
g1 = numpy.dot(X.T, dh)

print 'W2'
i = 0
while i < 3:
    old = W2[i][0]

    W2[i][0] = old + EPS
    hh = 1.0 / (1.0 + numpy.exp(-(numpy.dot(X, W1) + b1)))
    oo = 1.0 / (1.0 + numpy.exp(-(numpy.dot(hh, W2) + b2)))
    up = ((oo - y)**2).mean()

    W2[i][0] = old - EPS
    hh = 1.0 / (1.0 + numpy.exp(-(numpy.dot(X, W1) + b1)))
    oo = 1.0 / (1.0 + numpy.exp(-(numpy.dot(hh, W2) + b2)))
    dn = ((oo - y)**2).mean()

    W2[i][0] = old
    num = (up - dn) / (2*EPS)
    print i, g2[i][0], num, g2[i][0]/num
    i = i + 1

print 'W1'
i = 0
while i < 2:
    j = 0
    while j < 3:
        old = W1[i][j]

        W1[i][j] = old + EPS
        hh = 1.0 / (1.0 + numpy.exp(-(numpy.dot(X, W1) + b1)))
        oo = 1.0 / (1.0 + numpy.exp(-(numpy.dot(hh, W2) + b2)))
        up = ((oo - y)**2).mean()

        W1[i][j] = old - EPS
        hh = 1.0 / (1.0 + numpy.exp(-(numpy.dot(X, W1) + b1)))
        oo = 1.0 / (1.0 + numpy.exp(-(numpy.dot(hh, W2) + b2)))
        dn = ((oo - y)**2).mean()

        W1[i][j] = old
        num = (up - dn) / (2*EPS)
        print i, j, g1[i][j], num, g1[i][j]/num
        j = j + 1
    i = i + 1
