import numpy
from nn2 import Net

# same idea as check.py but against nn2, because i dont trust myself
# to have got the backwards loop right either

EPS = 0.00001

X = numpy.array([[0, 0], [0, 1], [1, 0], [1, 1]])
y = numpy.array([[0], [1], [1], [0]])

n = Net([2, 4, 3, 1], 5)

# my gradients, copied out of nn2.train
a = n.fwd(X)
d = (a[n.n] - y) * a[n.n] * (1 - a[n.n])
g = []
k = 0
while k < n.n:
    g.append(None)
    k = k + 1
i = n.n - 1
while i >= 0:
    g[i] = numpy.dot(a[i].T, d)
    if i > 0:
        d = numpy.dot(d, n.W[i].T) * a[i] * (1 - a[i])
    i = i - 1

worst = 0.0
L = 0
while L < n.n:
    i = 0
    while i < n.W[L].shape[0]:
        j = 0
        while j < n.W[L].shape[1]:
            old = n.W[L][i][j]
            n.W[L][i][j] = old + EPS
            up = ((n.go(X) - y)**2).mean()
            n.W[L][i][j] = old - EPS
            dn = ((n.go(X) - y)**2).mean()
            n.W[L][i][j] = old
            num = (up - dn) / (2*EPS)
            if abs(num) > 0.000000001:
                rel = abs(g[L][i][j] - num) / abs(num)
                if rel > worst:
                    worst = rel
            j = j + 1
        i = i + 1
    L = L + 1
print worst
