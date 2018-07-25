from numpy import conjugate as con
from numpy import kron
import numpy as np
import pprint
import basis
# import pandas as pd
import matplotlib.pyplot as plt
import random
import time


class density_matrix:
    bell = [1/np.sqrt(2)*np.array((1, 0, 0, 1)),
            1/np.sqrt(2)*np.array((1, 0, 0, -1)),
            1/np.sqrt(2)*np.array((0, 1, 1, 0)),
            1/np.sqrt(2)*np.array((0, 1, -1, 0))]
    initial = {"phy_plus": [0, 0.99], "phy_minus": [1, 0.01],
               "psi_plus": [2, 0], "psi_minus": [3, 0]}
    # latter one in [0, 1](1 is latter one) is the rate of the state.
    I = np.array(((1, 0), (0, 1)))
    x = np.array(((0, 1), (1, 0)))
    y = np.array(((0, -1j), (1j, 0)))
    z = np.array(((1, 0), (0, -1)))
    state_list = [I, x, y, z]

    @classmethod
    def two_tomography(self):
        bell = self.bell
        initial = self.initial
        state_list = self.state_list
        mk_rho = 0
        # state = ["I", "x", "y", "z"]
        t_rho = 0
        measurement_basis = basis.table_mb
        for i in initial.values():  # iterate for each bell pair
            probability = []
            Stokes_parameters = []
            rho = 0

            # calculate probability of each state.
            for measure in measurement_basis:
                    prob = np.vdot(np.dot(con(bell[i[0]]), con(measure)),
                                   np.dot(measure, bell[i[0]]))
                    # con is the conjugate operation
                    probability.append(prob * i[0])
            probability = np.array(probability).reshape(6, 6)
            # print(probability)

            # calculate stokes parameters
            S00 = probability[0][0] + probability[0][1]
            + probability[1][0] + probability[1][1]
            Stokes_parameters.append(S00)

            for c in range(0, 5, 2):
                Stokes_parameters.append(probability[c][c] -
                                         probability[c][c + 1] +
                                         probability[c + 1][c] -
                                         probability[c + 1][c + 1])

            for col in range(0, 5, 2):
                for low in range(0, 5, 2):
                    Stokes_parameters.append(probability[col][low] +
                                             probability[col + 1][low + 1] -
                                             probability[col][low + 1] -
                                             probability[col + 1][low])

            S10 = (probability[0][0] -
                   probability[0][1] +
                   probability[1][0] -
                   probability[1][1])
            S20 = (probability[2][2] -
                   probability[2][3] +
                   probability[3][2] -
                   probability[3][3])
            S30 = (probability[4][4] -
                   probability[4][5] +
                   probability[5][4] -
                   probability[5][5])

            Stokes_parameters.insert(4, S10)
            Stokes_parameters.insert(8, S20)
            Stokes_parameters.insert(12, S30)  # FIXME can more efficient
            # finished calculattion of Stokes_parameters

            # stat = [str(k) + str(c) for k in state for c in state]
            # making density matrix from stokes parameters
            krons = [kron(ul, rl) for ul in state_list for rl in state_list]
            # print(krons)
            for tl in range(len(Stokes_parameters)):
                rho = rho + Stokes_parameters[tl] * krons[tl]
            rho = 1 / 4 * rho
            # print(rho)

            mk_rho = mk_rho + rho

        for t in initial.values():
            t_rho = t_rho + t[1] * kron(bell[t[0]], con(bell[t[0]]))
        t_rho = np.array(t_rho, dtype="complex")

        # self.return_table(probability)
        # FIXME return value is not list but matrix. must repair
        # print("========== measure density matrix ==========")
        # pprint.pprint(mk_rho)
        # print("============ true density matrix ===========")
        # pprint.pprint(t_rho.reshape(4, 4))
        return mk_rho, t_rho, probability

    @classmethod
    def return_table(self, probability):
        # initial = self.initial
        # measurement_basis = basis.table_mb

        # print("==============probability table==============")
        """df = pd.DataFrame({'measurement_basis': ['X+', 'X-',
                                                'Y+', 'Y-',
                                                 'Z+', 'Z-'],
                            'X+':probability[0:36:6],
                            'X-':probability[1:36:6],
                            'Y+':probability[2:36:6],
                            'Y-':probability[3:36:6],
                            'Z+':probability[4:36:6],
                            'Z-':probability[5:36:6]})"""
        # print(df)
        # return df

    @classmethod
    def evaluation_density(self, iter):
        a, t_rho, p = self.two_tomography()
        rho = self.tomograph_density(iter)
        t_rho = np.array(t_rho).reshape(4, 4)
        rho = np.array(rho).reshape(4, 4)
        # print("=================result=========================")
        # print(t_rho.reshape(4, 4))
        # print(rho)
        # print("===========================================")
        error = 0
        for i in range(4):
            for j in range(4):
                error = error + np.square(t_rho[i][j] -
                                          rho[i][j]) / 2 * t_rho[i][j]
        print("===============differences===================")
        print(error)
        return error, rho, t_rho

    @classmethod
    def iterate_sim(self, iter):
        errors = []
        for k in range(iter):
            print("now", k)
            err, rho, t_rho = self.evaluation_density(k)
            errors.append(err)
        print("==============tomography_matrix==============")
        pprint.pprint(rho)
        print("=============true density matrix=============")
        pprint.pprint(t_rho)
        plt.plot(errors)
        plt.show()

    @classmethod
    def tomograph_density(self, iter):
        state_list = self.state_list
        initial = self.initial
        bell = self.bell
        to_rho = 0
        measurement_basis = basis.table_mb
        for j in initial.values():
            probability = []
            st = []  # stokes parameter
            rho = 0
            tomography = np.array(((0, 0, 0, 0, 0, 0),
                                   (0, 0, 0, 0, 0, 0),
                                   (0, 0, 0, 0, 0, 0),
                                   (0, 0, 0, 0, 0, 0),
                                   (0, 0, 0, 0, 0, 0),
                                   (0, 0, 0, 0, 0, 0)),
                                  dtype='complex')  # FIXME zeros does not work

            result = np.array(((0, 0, 0, 0, 0, 0),
                               (0, 0, 0, 0, 0, 0),
                               (0, 0, 0, 0, 0, 0),
                               (0, 0, 0, 0, 0, 0),
                               (0, 0, 0, 0, 0, 0),
                               (0, 0, 0, 0, 0, 0)),
                              dtype='complex')
            sum = 0
            tomography.reshape(6, 6)
            for n in range(iter):
                ran = random.randint(0, 5)
                ran2 = random.randint(0, 5)
                tomography[ran][ran2] = tomography[ran][ran2] + 1

            for measure in measurement_basis:
                    prob = np.vdot(np.dot(con(bell[j[0]]), con(measure)),
                                   np.dot(measure, bell[j[0]]))
                    # con is the conjugate operation
                    probability.append(prob)
            probability = np.array(probability).reshape(6, 6)

            for cols in range(6):
                for lows in range(6):
                    result[cols][lows] = (probability[cols][lows] *
                                          tomography[cols][lows])
            sum = result[0][0] + result[0][1] + result[1][0] + result[1][1]

            for cols in range(6):
                for lows in range(6):
                    if sum == 0:
                        result[cols][lows] = result[cols][lows] / 1
                    else:
                        result[cols][lows] = result[cols][lows] * j[1] / sum

            S00 = result[0][0] + result[0][1] + result[1][0] + result[1][1]
            st.append(S00)

            for c in range(0, 5, 2):
                st.append(result[c][c] - result[c][c + 1] +
                          result[c + 1][c] - result[c + 1][c + 1])

            for col in range(0, 5, 2):
                for low in range(0, 5, 2):
                    # print(col, low)
                    # print(re[col][low] + re[col + 1][low + 1] +
                    # re[col][low + 1] + re[col + 1][low])
                    st.append(result[col][low] + result[col + 1][low + 1] -
                              result[col][low + 1] - result[col + 1][low])

            S10 = result[0][0] - result[0][1] + result[1][0] - result[1][1]
            S20 = result[2][2] - result[2][3] + result[3][2] - result[3][3]
            S30 = result[4][4] - result[4][5] + result[5][4] - result[5][5]

            st.insert(4, S10)
            st.insert(8, S20)
            st.insert(12, S30)

            # making density matrix from stokes parameters
            krons = [kron(ul, rl) for ul in state_list for rl in state_list]
            for tl in range(len(st)):
                rho = rho + st[tl] * krons[tl]
            rho = 1 / 4 * rho

            to_rho = to_rho + rho

        # print("========== measure density matrix ==========")
        # pprint.pprint(to_rho)

        return to_rho

if __name__ == '__main__':
    start = time.time()
    density_matrix.iterate_sim(1000)
    # difine how many times iterate in the method iterate_sim(iteration number)
    print(time.time() - start)
    # density_matrix.two_tomography()
