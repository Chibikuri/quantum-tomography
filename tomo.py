import numpy as np
from numpy import conjugate as con
from numpy import kron
import pprint
import basis
import pandas as pd
import matplotlib.pyplot as plt
import random

class density_matrix:
    # initial = 0.2 * 1 / np.sqrt(2) * np.array((1, 0, 0, 1)) + 0.1 * 1 /np.sqrt(2) * np.array((0, 1, 1, 0)) + 0.1 * 1/np.sqrt(2) * np.array((0, 1, 0, 1))
    #+ 0.1 * 1/np.sqrt(2) * np.array((1, 0, 1, 0)) + 0.2*np.array((1, 0, 0, 0)) + 0.4*np.array((0, 0, 0, 1))
    #initial = 1/np.sqrt(2)*np.array((1, 0, 0, 1))

    # conjugate_initial = con(initial)

    def __init__(self):
        self.I = np.array(((1, 0), (0, 1)))
        self.x = np.array(((0, 1), (1, 0)))
        self.y = np.array(((0, -1j), (1j, 0)))
        self.z = np.array(((1, 0), (0, -1)))
        ran1 = np.randoint(0, 3)
        ran2 = np.randoint(0, 3)
        bell = [1/np.sqrt(2)*np.array((1, 0, 0, 1)), 1/np.sqrt(2)*np.array((1, 0, 0, -1)),
                1/np.sqrt(2)*np.array((0, 1, 1, 0)), 1/np.sqrt(2)*np.array((0, 1, -1, 0))]
        self.initial = 0.8*bell[ran1] + 0.2*bell[ran2]
        self.con_initial = con(self.initial)


    def two_tomography(self):
        Stokes_parameters = []
        probability = []
        rho = 0
        err = 0.4 * np.array((0, 1, 1, 0))
        # initial = self.initial
        # con_initial = self.conjugate_initial
        e_initial = self.initial - err
        e_con_initial = con(e_initial)
        I = self.I
        x = self.x
        y = self.y
        z = self.z
        state_list = [I, x, y, z]

        mxp = 1/2*(I+x)
        mxn = 1/2*(I-x)
        myp = 1/2*(I+y)
        myn = 1/2*(I-y)
        mzp = 1/2*(I+z)
        mzn = 1/2*(I-z)

        #calculate probability of each state.
        measurement_basis = basis.mb
        for measure in measurement_basis:
            prob = np.vdot(np.dot(e_con_initial, con(measure)), np.dot(measure, e_initial))
            probability.append(prob)

        #calculate stokes parameters
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

        #making density matrix from stokes parameters
        krons = [kron(ul, rl) for ul in state_list for rl in state_list]
        for tl in range(len(Stokes_parameters)):
            rho = rho + Stokes_parameters[tl] * krons[tl]
        rho = 1 / 4 * rho
        t_rho = kron(initial, con_initial).reshape(4, 4)

        #return resule
        self.return_table()

        print("========== measure density matrix ==========")
        pprint.pprint(rho)
        print("============ true density matrix ===========")
        pprint.pprint(t_rho)
        return rho, t_rho


    def return_table(self):
        measurement_basis = basis.table_mb
        probability = []
        for measure in measurement_basis:
            prob = np.vdot(np.dot(self.con_initial, con(measure)), np.dot(measure, self.initial))
            probability.append(prob)
        print("==============probability table==============")
        df = pd.DataFrame({'measurement_basis':['X+', 'X-', 'Y+', 'Y-', 'Z+', 'Z-'],
                           'X+':probability[0:36:6],
                           'X-':probability[1:36:6],
                           'Y+':probability[2:36:6],
                           'Y-':probability[3:36:6],
                           'Z+':probability[4:36:6],
                           'Z-':probability[5:36:6]})
        print(df)

    def evaluation_density(self):
        error = 0
        rho, t_rho = self.two_tomography()
        for i in range(4):
            for j in range(4):
                error = error + np.square(t_rho[i, j] - rho[i, j]) / 2 * t_rho[i, j]
        print("===============defferences===================")
        print(error)
        return error

    @classmethod
    def iterate_sim(self):
        errors = []
        n = 1000
        for k in range(n):
            e_rate = n / 100000
            err = self.evaluation_density(e_rate)
            errors.append(err)
        plt.plot(errors)
        plt.show()



if __name__ == '__main__':
    density_matrix.iterate_sim()
