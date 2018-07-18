import numpy as np
from numpy import conjugate as con
from numpy import kron
import pprint
import basis
import pandas as pd

class density_matrix:
    #initial = 0.1 * 1 / np.sqrt(2) * np.array((1, 0, 0, 1)) + 0.1 * 1 /np.sqrt(2) * np.array((0, 1, 1, 0)) + 0.8 * 1/np.sqrt(2) * np.array((0, 1, 0, 1))
    initial = 1/np.sqrt(2)*np.array((1, 0, 0, 1))
    conjugate_initial = con(initial)
    I = np.array(((1, 0), (0, 1)))
    x = np.array(((0, 1), (1, 0)))
    y = np.array(((0, -1j), (1j, 0)))
    z = np.array(((1, 0), (0, -1)))
    @classmethod
    def two_tomography(self):
        Stokes_parameters = []
        probability = []
        initial = self.initial
        con_initial = self.conjugate_initial
        I = self.I
        x = self.x
        y = self.y
        z = self.z

        mxp = 1/2*(I+x)
        mxn = 1/2*(I-x)
        myp = 1/2*(I+y)
        myn = 1/2*(I-y)
        mzp = 1/2*(I+z)
        mzn = 1/2*(I-z)

        measurement_basis = basis.mb
        for measure in measurement_basis:
            prob = np.vdot(np.dot(con_initial, con(measure)), np.dot(measure, initial))
            probability.append(prob)

        S00 = probability[0] + probability[1] + probability[2] + probability[3]
        
        Stokes_parameters.append(S00)
        for sp0 in range(0, len(probability), 16):
            S = probability[sp0] - probability[sp0 + 1] + probability[sp0 + 2] - probability[sp0 + 3]
            Stokes_parameters.append(S)

        for j in range(0, len(measurement_basis), 4):
            S = probability[j] - probability[j + 1] - probability[j + 2] + probability[j + 3]
            Stokes_parameters.append(S)

        S10 = probability[0] + probability[1] - probability[2] - probability[3]
        S20 = probability[16] + probability[17] - probability[18] - probability[19]
        S30 = probability[32] + probability[33] - probability[34] - probability[35]
        st_l = [S10, S20, S30]
        Stokes_parameters.insert(4, S10)
        Stokes_parameters.insert(8, S20)
        Stokes_parameters.insert(12, S30)

        state_list = [I, x, y, z]
        rho = 0
        krons = [kron(ul, rl) for ul in state_list for rl in state_list]

        for tl in range(len(Stokes_parameters)):
            rho = rho + Stokes_parameters[tl] * krons[tl]
        rho = 1/4 * rho

        t_rho = kron(initial, con_initial).reshape(4, 4)

        print("========== measure density matrix ==========")
        pprint.pprint(rho)
        print("========== true density matrix ==========")
        pprint.pprint(t_rho)
        return probability


    @classmethod
    def return_table(self):
        initial = self.initial
        con_initial = self.conjugate_initial
        measurement_basis = basis.table_mb
        probability = []
        for measure in measurement_basis:
            prob = np.vdot(np.dot(con_initial, con(measure)), np.dot(measure, initial))
            probability.append(prob)
        print("==========probability table==========")
        df = pd.DataFrame({'measurement_basis':['X+', 'X-', 'Y+', 'Y-', 'Z+', 'Z-'],
                           'X+':probability[0:36:6],
                           'X-':probability[1:36:6],
                           'Y+':probability[2:36:6],
                           'Y-':probability[3:36:6],
                           'Z+':probability[4:36:6],
                           'Z-':probability[5:36:6]})
        print(df)


if __name__ == '__main__':
    density_matrix.return_table()
    density_matrix.two_tomography()
