import numpy

# same as test.py but any number of layers of any size so i can point it
# at mnist. n = Net([784, 100, 10])

class Net:

  def __init__(self, sizes, seed=1):
    r = numpy.random.RandomState(seed)
    self.W = []
    self.b = []
    self.n = len(sizes) - 1
    i = 0
    while i < self.n:
      # has to depend on the layer size, learned that the hard way in the
      # theano one
      z = 1.0 / numpy.sqrt(sizes[i])
      self.W.append(r.uniform(-z, z, (sizes[i], sizes[i+1])))
      self.b.append(numpy.zeros((1, sizes[i+1])))
      i = i + 1

  def fwd(self, X):
    a = [X]
    i = 0
    while i < self.n:
      a.append(1.0 / (1.0 + numpy.exp(-(numpy.dot(a[i], self.W[i]) + self.b[i]))))
      i = i + 1
    return a

  def train(self, X, y, lr):
    a = self.fwd(X)
    d = (a[self.n] - y) * a[self.n] * (1 - a[self.n])
    i = self.n - 1
    while i >= 0:
      gW = numpy.dot(a[i].T, d)
      gb = d.sum(axis=0)
      if i > 0:
        d = numpy.dot(d, self.W[i].T) * a[i] * (1 - a[i])
      self.W[i] = self.W[i] - lr * gW
      self.b[i] = self.b[i] - lr * gb
      i = i - 1
    return ((a[self.n] - y)**2).mean()

  def go(self, X):
    a = self.fwd(X)
    return a[self.n]
