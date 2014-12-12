nn from scratch
===============

a neural net with no libraries, numpy only for the matrix multiplies. i
wanted to understand backprop instead of letting theano do it.

test.py      2 in 3 hidden 1 out, learns xor, everything written out
check.py     my derivatives vs the numerical ones. RUN THIS FIRST
nn2.py       same maths, any number of layers
check2.py    same check against nn2
mnist.py     points nn2 at mnist.pkl.gz, gets about 8.9%
thing.py     writes the first layer out as text so you can look at it

test_old.py is the version with the PLUS in it, dont run it, its kept
because it took three weeks off me.

notes2.txt is the sign thing.

paths are hardcoded to my desktop. python 2.7, needs numpy.
