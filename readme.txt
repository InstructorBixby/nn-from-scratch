nn from scratch
===============

a neural net with no libraries, numpy only for the matrix multiplies.
written in school to understand backprop instead of letting theano do it.

test.py      2 in 3 hidden 1 out, learns xor, everything written out
check.py     my derivatives vs the numerical ones
nn2.py       same maths, any number of layers
check2.py    same check against nn2
mnist.py     points nn2 at mnist.pkl.gz, gets about 8.9%
thing.py     writes the first layer out as text

test_old.py is the version with the PLUS in it. kept because it took
three weeks off me. notes2.txt is that story.

state
-----

half ported to python 3 after py2 went end of life. test.py, check.py
and nn2.py run and give the same numbers they always did. mnist.py,
check2.py and thing.py are still python 2 and im not fixing them, the
pickle they want is a dead link and anyone who wants a working mlp has
four libraries to pick from now.

the maths in notes.txt was always fine. it was the sign.

needs numpy.
