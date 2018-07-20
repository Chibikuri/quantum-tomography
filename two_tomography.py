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
    # + 0.1 * 1/np.sqrt(2) * np.array((1, 0, 1, 0)) + 0.2*np.array((1, 0, 0, 0)) + 0.4*np.array((0, 0, 0, 1))
    initial = 1/np.sqrt(2)*np.array((1, 0, 0, 1))
    bell = [1/np.sqrt(2)*np.array((1, 0, 0, 1)), 1/np.sqrt(2)*np.array((1, 0, 0, -1)),
            1/np.sqrt(2)*np.array((0, 1, 1, 0)), 1/np.sqrt(2)*np.array((0, 1, -1, 0))]
    con_initial = con(initial)
    I = np.array(((1, 0), (0, 1)))
    x = np.array(((0, 1), (1, 0)))
    y = np.array(((0, -1j), (1j, 0)))
    z = np.array(((1, 0), (0, -1)))
    state_list = [I, x, y, z]
    ran  = random.randint(0, 3)
    @classmethod
    def two_tomography(self, error_rate):
        initial = self.initial
        con_initial = self.con_initial
        Stokes_parameters = []
        state = ["I", "x", "y", "z"]
        probability = []
        rho = 0
        err = error_rate * np.array((0, 1, 1, 0))
        e_initial = initial - err
        e_con_initial = con(e_initial)
        I = self.I
        x = self.x
        y = self.y
        z = self.z
        state_list = self.state_list


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
        print(Stokes_parameters)
        stat = [str(k) + str(c) for k in state for c in state]
        print(len(stat))


        #making density matrix from stokes parameters
        krons = [kron(ul, rl) for ul in state_list for rl in state_list]
        # print(krons)
        for tl in range(len(Stokes_parameters)):
            rho = rho + Stokes_parameters[tl] * krons[tl]
        rho = 1 / 4 * rho
        t_rho = kron(initial, con_initial).reshape(4, 4)

        #return resule
        self.return_table(error_rate)

        # print("========== measure density matrix ==========")
        # pprint.pprint(rho)
        # print("============ true density matrix ===========")
        # pprint.pprint(t_rho)
        return rho, t_rho, probability


    @classmethod
    def return_table(self, error_rate):
        initial = self.initial
        con_initial = self.con_initial
        err = error_rate * np.array((0, 1, 1, 0))
        # initial = self.initial
        # con_initial = self.conjugate_initial
        e_initial = initial - err
        e_con_initial = con(e_initial)
        measurement_basis = basis.table_mb
        probability = []
        for measure in measurement_basis:
            prob = np.vdot(np.dot(e_con_initial, con(measure)), np.dot(measure, e_initial))
            probability.append(prob)

        print("==============probability table==============")
        df = pd.DataFrame({'measurement_basis':['X+', 'X-', 'Y+', 'Y-', 'Z+', 'Z-'],
                           'X+':probability[0:36:6],
                           'X-':probability[1:36:6],
                           'Y+':probability[2:36:6],
                           'Y-':probability[3:36:6],
                           'Z+':probability[4:36:6],
                           'Z-':probability[5:36:6]})
        # print(df.iloc[[0, 1], [0, 1, 2]])
        # print(df)
        return df, probability

    @classmethod
    def evaluation_density(self, err, iter):
        error = 0
        rho, t_rho = self.tomography_simulater(err, iter)
        for i in range(4):
            for j in range(4):
                error = error + np.square(t_rho[i, j] - rho[i, j]) / 2 * t_rho[i, j]
        print("===============defferences===================")
        print(error)
        return error

    @classmethod
    def iterate_sim(self, e_rate):
        errors = []
        n = 1000
        for k in range(n):
            err = self.evaluation_density(e_rate, k)
            errors.append(err)
        plt.plot(errors)
        plt.show()

    @classmethod
    def tomography_simulater(self, err, iter):
        tomography = np.array(((0, 0, 0, 0, 0, 0),
                               (0, 0, 0, 0, 0, 0),
                               (0, 0, 0, 0, 0, 0),
                               (0, 0, 0, 0, 0, 0),
                               (0, 0, 0, 0, 0, 0),
                               (0, 0, 0, 0, 0, 0)), dtype = 'complex') #zerosが上手く動かない
        result = np.array(((0, 0, 0, 0, 0, 0),
                           (0, 0, 0, 0, 0, 0),
                           (0, 0, 0, 0, 0, 0),
                           (0, 0, 0, 0, 0, 0),
                           (0, 0, 0, 0, 0, 0),
                           (0, 0, 0, 0, 0, 0)), dtype = 'complex')
        # print(tomography[0][0])
        e_initial = self.initial - err*1/np.sqrt(2)*np.array((1, 1, 1, 1))
        e_con_initial = con(e_initial)
        measurement_basis = basis.table_mb
        probability = []
        for measure in measurement_basis:
            prob = np.vdot(np.dot(e_con_initial, con(measure)), np.dot(measure, e_initial))
            probability.append(prob)

        probability = np.array(probability).reshape(6, 6)
        print("=========probabilitY=====================")
        # print(probability)
        sum = 0
        tomography.reshape(6, 6)
        for n in range(iter):
            ran = random.randint(0, 5)
            ran2 = random.randint(0, 5)
            tomography[ran][ran2] = tomography[ran][ran2] + 1
            # print(tomography)
        # print(probability)
        for cols in range(6):
            for lows in range(6):
                result[cols][lows] = probability[cols][lows] * tomography[cols][lows]

        sum = result[0][0] + result[0][1] + result[1][0] + result[1][1]

        for cols in range(6):
            for lows in range(6):
                result[cols][lows] = result[cols][lows] / sum

        # print(result.reshape(6, 6))
        # print(result)
        # print("=====================")
        # print(sum)

        return self.return_real_density(result)



    @classmethod
    def return_real_density(self, re):
        st = []
        rho = 0
        t_rho = 0
        state_list = self.state_list
        initial = self.initial
        con_initial = self.con_initial
        S00 = re[0][0] + re[0][1] + re[1][0] + re[1][1]
        st.append(S00)
        # print(re[0][0])

        for c in range(0, 5, 2):
            st.append(re[c][c] - re[c][c + 1] + re[c + 1][c] - re[c + 1][c + 1])

        for col in range(0, 5, 2):
            for low in range(0, 5, 2):
                # print(col, low)
                # print(re[col][low] + re[col + 1][low + 1] + re[col][low + 1] + re[col + 1][low])
                st.append(re[col][low] + re[col + 1][low + 1] - re[col][low + 1] - re[col + 1][low])

        #print(st)
        S10 = re[0][0] - re[0][1] + re[1][0] - re[1][1]
        S20 = re[2][2] - re[2][3] + re[3][2] - re[3][3]
        S30 = re[4][4] - re[4][5] + re[5][4] - re[5][5]

        st.insert(4, S10)
        st.insert(8, S20)
        st.insert(12, S30)
        print(st)

        #making density matrix from stokes parameters
        krons = [kron(ul, rl) for ul in state_list for rl in state_list]
        print("==========state statelsi =================")
        # print(state_list)
        for tl in range(len(st)):
            # print("==========================")
            rho = rho + st[tl] * krons[tl]
            # print(rho)
        rho = 1 / 4 * rho
        t_rho = kron(initial, con_initial).reshape(4, 4)

        # #return resule
        # self.return_table(error_rate)

        print("========== measure density matrix ==========")
        pprint.pprint(rho)
        print("============ true density matrix ===========")
        pprint.pprint(t_rho)
        return rho, t_rho

if __name__ == '__main__':
    density_matrix.iterate_sim(0.3)#e_rate
    # density_matrix.evaluation_density(0, 100000)
     # density_matrix.tomography_simulater(0, 100000)
     # print("=======================")
     # density_matrix.two_tomography(0)
