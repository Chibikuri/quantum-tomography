import numpy as np
from numpy import conjugate as con
from numpy import kron as kron
import pprint

class density_matrix:
    initial = 0.1 * 1 / np.sqrt(2) * np.array((1, 0, 0, 1)) + 0.1 * 1 /np.sqrt(2) * np.array((0, 1, 1, 0)) + 0.8 * 1/np.sqrt(2) * np.array((0, 1, 0, 1))
    I = np.array(((1, 0), (0, 1)))
    x = np.array(((0, 1), (1, 0)))
    y = np.array(((0, -1j), (1j, 0)))
    z = np.array(((1, 0), (0, -1)))
    @classmethod
    def two_tomography(self):
        initial = self.initial
        I = self.I
        x = self.x
        y = self.y
        z = self.z
        con_initial = con(initial)
        mxp = 1/2*(I+x)
        mxn = 1/2*(I-x)
        myp = 1/2*(I+y)
        myn = 1/2*(I-y)
        mzp = 1/2*(I+z)
        mzn = 1/2*(I-z)

#list of measurement basis of two qubits
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
#=============================
        measurement_basis = [mxpxp, mxpxn, mxnxp, mxnxn, mxpyp, mxpyn, mxnyp, mxnyn, mxpzp, mxpzn, mxnzp, mxnzn,
                             mypxp, mypxn, mynxp, mynxn, mypyp, mypyn, mynyp, mynyn, mypzp, mypzn, mynzp, mynzn,
                             mzpxp, mzpxn, mznxp, mznxn, mzpyp, mzpyn, mznyp, mznyn, mzpzp, mzpzn, mznzp, mznzn]

        probability = []
        for measure in measurement_basis:
            prob = np.vdot(np.dot(con_initial, con(measure)), np.dot(measure, initial))
            probability.append(prob)

        Stokes_parameters = []
        # S00 = probability[measurement_basis.index(mxpxp)] + probability[measurement_basis.index(mxpxn)] + probability[measurement_basis.index(mxnxp)] + probability[measurement_basis.index(mxnxn)]
        # S01 = probability[measurement_basis.index(mxpxp)] - probability[measurement_basis.index(mxpxn)] + probability[measurement_basis.index(mxnxp)] - probability[measurement_basis.index(mxnxn)]
        # S02 = probability[measurement_basis.index(mypyp)] - probability[measurement_basis.index(mypyn)] + probability[measurement_basis.index(mynyp)] - probability[measurement_basis.index(mynyn)]
        # S03 = probability[measurement_basis.index(mzpzp)] - probability[measurement_basis.index(mzpzn)] + probability[measurement_basis.index(mznzp)] - probability[measurement_basis.index(mznzn)]

        S00 = probability[0] + probability[1] + probability[2] + probability[3]
        S01 = probability[0] - probability[1] + probability[2] - probability[3]
        S02 = probability[16] - probability[17] + probability[18] - probability[19]
        S03 = probability[32] - probability[33] + probability[34] - probability[35]
        S10 = probability[0] + probability[1] - probability[2] - probability[3]
        S20 = probability[16] + probability[17] - probability[18] - probability[19]
        S30 = probability[32] + probability[33] - probability[34] - probability[35]

        Stokes_parameters = [S00, S01, S02, S03, S10, S20, S30]
        #print(S00)
        #print(S01)
        #print(S02)
        #print(S03)
        #print(Stokes_parameters)

        #XX, YY, ZZ
        for j in range(0, len(measurement_basis) - 3, 4):
            S = probability[j] - probability[j + 1] - probability[j + 2] + probability[j + 3]
            Stokes_parameters.append(S)
        print(Stokes_parameters)

        rho = 1/4*(Stokes_parameters[0] * kron(I, I)
                   + Stokes_parameters[1] * kron(I, x)
                   + Stokes_parameters[2] * kron(I, y)
                   + Stokes_parameters[3] * kron(I, z)
                   + Stokes_parameters[4] * kron(x, I)
                   + Stokes_parameters[5] * kron(y, I)
                   + Stokes_parameters[6] * kron(z, I)
                   + Stokes_parameters[7] * kron(x, x)
                   + Stokes_parameters[8] * kron(x, y)
                   + Stokes_parameters[9] * kron(x, z)
                   + Stokes_parameters[10] * kron(y, x)
                   + Stokes_parameters[11] * kron(y, y)
                   + Stokes_parameters[12] * kron(y, z)
                   + Stokes_parameters[13] * kron(z, x)
                   + Stokes_parameters[14] * kron(z, y)
                   + Stokes_parameters[15] * kron(z, z))



        t_rho = kron(initial, con_initial).reshape(4, 4)
        print("========== measure density matrix ==========")
        pprint.pprint(rho)
        print("========== true density matrix ==========")
        pprint.pprint(t_rho)




if __name__ == '__main__':
    density_matrix.two_tomography()
