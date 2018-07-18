from numpy import kron
import numpy as np

I = np.array(((1, 0), (0, 1)))
x = np.array(((0, 1), (1, 0)))
y = np.array(((0, -1j), (1j, 0)))
z = np.array(((1, 0), (0, -1)))

mxp = 1/2*(I+x)
mxn = 1/2*(I-x)
myp = 1/2*(I+y)
myn = 1/2*(I-y)
mzp = 1/2*(I+z)
mzn = 1/2*(I-z)

mxpxp = kron(mxp, mxp)#, 0)
mxpxn = kron(mxp, mxn)#, 0)
mxnxp = kron(mxn, mxp)#, 0)
mxnxn = kron(mxn, mxn)#, 0)

mxpyp = kron(mxp, myp)#, 0)
mxpyn = kron(mxp, myn)#, 0)
mxnyp = kron(mxn, myp)#, 0)
mxnyn = kron(mxn, myn)#, 0)

mxpzp = kron(mxp, mzp)#, 0)
mxpzn = kron(mxp, mzn)#, 0)
mxnzp = kron(mxn, mzp)#, 0)
mxnzn = kron(mxn, mzn)#, 0)
#===============================
mypxp = kron(myp, mxp)#, 0)
mypxn = kron(myp, mxn)#, 0)
mynxp = kron(myn, mxp)#, 0)
mynxn = kron(myn, mxn)#, 0)

mypyp = kron(myp, myp)#, 0)
mypyn = kron(myp, myn)#, 0)
mynyp = kron(myn, myp)#, 0)
mynyn = kron(myn, myn)#, 0)

mypzp = kron(myp, mzp)#, 0)
mypzn = kron(myp, mzn)#, 0)
mynzp = kron(myn, mzp)#, 0)
mynzn = kron(myn, mzn)#, 0)
#================================
mzpxp = kron(mzp, mxp)#, 0)
mzpxn = kron(mzp, mxn)#, 0)
mznxp = kron(mzn, mxp)#, 0)
mznxn = kron(mzn, mxn)#, 0)

mzpyp = kron(mzp, myp)#, 0)
mzpyn = kron(mzp, myn)#, 0)
mznyp = kron(mzn, myp)#, 0)
mznyn = kron(mzn, myn)#, 0)

mzpzp = kron(mzp, mzp)#, 0)
mzpzn = kron(mzp, mzn)#, 0)
mznzp = kron(mzn, mzp)#, 0)
mznzn = kron(mzn, mzn)#, 0)

mb = [mxpxp, mxpxn, mxnxp, mxnxn, mxpyp, mxpyn, mxnyp, mxnyn, mxpzp, mxpzn, mxnzp, mxnzn,
      mypxp, mypxn, mynxp, mynxn, mypyp, mypyn, mynyp, mynyn, mypzp, mypzn, mynzp, mynzn,
      mzpxp, mzpxn, mznxp, mznxn, mzpyp, mzpyn, mznyp, mznyn, mzpzp, mzpzn, mznzp, mznzn]

table_mb = [mxpxp, mxpxn, mxpyp, mxpyn, mxpzp, mxpzn,
            mxnxp, mxnxn, mxnyp, mxnyn, mxnzp, mxnzn,
            mypxp, mypxn, mypyp, mypyn, mypzp, mypzn,
            mynxp, mynxn, mynyp, mynyn, mynzp, mynzn,
            mzpxp, mzpxn, mzpyp, mzpyn, mzpzp, mzpzn,
            mznxp, mznxn, mznyp, mznyn, mznzp, mznzn]
