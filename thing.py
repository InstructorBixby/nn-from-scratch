import numpy
from nn2 import Net

# write the first layer out as text so i can stare at it. 784x100 is
# too much to print so just the first 20 hidden units

n = Net([784, 100, 10])
f = open('/home/panic/Desktop/nn/weights.txt', 'w')
j = 0
while j < 20:
    c = n.W[0][:, j]
    f.write('unit ' + str(j) + ' ' + str(c.min()) + ' ' + str(c.max()) + '\n')
    i = 0
    while i < 784:
        k = i
        line = ''
        while k < i + 28:
            line = line + '%+.3f ' % c[k]
            k = k + 1
        f.write(line + '\n')
        i = i + 28
    f.write('\n')
    j = j + 1
f.close()
print 'done'
