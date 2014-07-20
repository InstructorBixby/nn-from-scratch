import numpy

# what does the sigmoid derivative actually look like

x = -6.0
while x < 6.0:
    s = 1.0 / (1.0 + numpy.exp(-x))
    print x, s, s*(1-s)
    x = x + 0.5

# its basically zero past about 5 either way. so if the weights are
# big everything stops moving. thats the thing people call saturated
