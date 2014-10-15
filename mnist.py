import gzip
import cPickle
import numpy
from nn2 import Net

# same mnist.pkl.gz as the theano repo

LR = 0.5
B = 10
HID = 100

f = gzip.open('/home/panic/Desktop/mnist/mnist.pkl.gz', 'rb')
d = cPickle.load(f)
f.close()
trX = d[0][0]
trY = d[0][1]
teX = d[2][0]
teY = d[2][1]

# labels are 0..9 and the net has 10 outputs so one hot them
T = numpy.zeros((len(trY), 10))
i = 0
while i < len(trY):
    T[i][trY[i]] = 1.0
    i = i + 1

n = Net([784, HID, 10])
nb = len(trX) / B
rr = numpy.random.RandomState(99)

e = 0
while e < 30:
    # shuffle or it sees the same batches in the same order every
    # time and the error goes down in stairs
    o = rr.permutation(len(trX))
    sX = trX[o]
    sT = T[o]
    i = 0
    while i < nb:
        n.train(sX[i*B:i*B+B], sT[i*B:i*B+B], LR)
        i = i + 1
    pr = n.go(teX).argmax(axis=1)
    print e, (pr != teY).mean()
    e = e + 1
print "DONE"
