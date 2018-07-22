import numpy as np
from numpy import conjugate as con
from numpy import kron
import pprint
import basis
import pandas as pd
import matplotlib.pyplot as plt
import random
import cmath

class density_matrix:
    bell = [1/np.sqrt(2)*np.array((1, 0, 0, 1)), 1/np.sqrt(2)*np.array((1, 0, 0, -1)),
            1/np.sqrt(2)*np.array((0, 1, 1, 0)), 1/np.sqrt(2)*np.array((0, 1, -1, 0))]
    initial = {"phy_plus":[0, 0.6], "phy_minus":[1, 0], "psi_plus":[2, 0], "psi_minus":[3, 0]} #latter one in [0, 1](1 is latter one) is the rate of the state.
    I = np.array(((1, 0), (0, 1)))
    x = np.array(((0, 1), (1, 0)))
    y = np.array(((0, -1j), (1j, 0)))
    z = np.array(((1, 0), (0, -1)))
    state_list = [I, x, y, z]
    @classmethod
    def two_tomography(self, re):
        bell = self.bell
        initial = self.initial
        state_list = self.state_list
        I = self.I
        x = self.x
        y = self.y
        z = self.z
        mk_rho = 0
        state = ["I", "x", "y", "z"]
        t_rho = 0
        measurement_basis = basis.table_mb
        for i in initial.values():#iterate for each bell pair
            probability = []
            Stokes_parameters = []
            rho = 0

            #calculate probability of each state.
            for measure in measurement_basis:
                    prob = np.vdot(np.dot(con(i[1] * bell[i[0]]), con(measure)), np.dot(measure, i[1] * bell[i[0]]))  #con is the conjugate operation
                    probability.append(prob)
            probability =  np.array(probability).reshape(6, 6)
            print(probability)

            #calculate stokes parameters
            S00 = probability[0][0] + probability[0][1] + probability[1][0] + probability[1][1]
            Stokes_parameters.append(S00)

            for c in range(0, 5, 2):
                Stokes_parameters.append(probability[c][c] - probability[c][c + 1] + probability[c + 1][c] - probability[c + 1][c + 1])
                print(Stokes_parameters)

            for col in range(0, 5, 2):
                for low in range(0, 5, 2):
                    Stokes_parameters.append(probability[col][low] + probability[col + 1][low + 1] - probability[col][low + 1] - probability[col + 1][low])
                    print(Stokes_parameters)
            S10 = probability[0][0] - probability[0][1] + probability[1][0] - probability[1][1]
            S20 = probability[2][2] - probability[2][3] + probability[3][2] - probability[3][3]
            S30 = probability[4][4] - probability[4][5] + probability[5][4] - probability[5][5]

            Stokes_parameters.insert(4, S10)
            Stokes_parameters.insert(8, S20)
            Stokes_parameters.insert(12, S30) #FIXME can more efficient
            #finished calculattion of Stokes_parameters

            stat = [str(k) + str(c) for k in state for c in state]
            #making density matrix from stokes parameters
            krons = [kron(ul, rl) for ul in state_list for rl in state_list]
            # print(krons)
            for tl in range(len(Stokes_parameters)):
                rho = rho + Stokes_parameters[tl] * krons[tl]
            rho = 1 / 4 * rho
            print(rho)

            mk_rho = mk_rho + rho

        for t in initial.values():
            t_rho = t_rho + kron(t[1] * bell[t[0]], con(t[1] * bell[t[0]]))

        # self.return_table(probability) #FIXME return value is not list but matrix. must repair
        print("========== measure density matrix ==========")
        pprint.pprint(mk_rho)
        print("============ true density matrix ===========")
        pprint.pprint(t_rho.reshape(4, 4))
        return mk_rho, t_rho, probability


    @classmethod
    def return_table(self, probability):
        initial = self.initial
        measurement_basis = basis.table_mb

        print("==============probability table==============")
        df = pd.DataFrame({'measurement_basis':['X+', 'X-', 'Y+', 'Y-', 'Z+', 'Z-'],
                           'X+':probability[0:36:6],
                           'X-':probability[1:36:6],
                           'Y+':probability[2:36:6],
                           'Y-':probability[3:36:6],
                           'Z+':probability[4:36:6],
                           'Z-':probability[5:36:6]})
        print(df)
        return df

    @classmethod
    def evaluation_density(self, iter):
        rho, t_rho = self.tomography_simulater(iter)
        for i in range(4):
            for j in range(4):
                error = error + np.square(t_rho[i, j] - rho[i, j]) / 2 * t_rho[i, j]
        print("===============differences===================")
        print(error)
        return error

    @classmethod
    def iterate_sim(self):
        errors = []
        n = 1000
        for k in range(n):
            err = self.evaluation_density(k)
            errors.append(err)
        plt.plot(errors)
        plt.show()

    @classmethod
    def tomography_simulater(self, iter):
        tomography = np.array(((0, 0, 0, 0, 0, 0),
                               (0, 0, 0, 0, 0, 0),
                               (0, 0, 0, 0, 0, 0),
                               (0, 0, 0, 0, 0, 0),
                               (0, 0, 0, 0, 0, 0),
                               (0, 0, 0, 0, 0, 0)), dtype = 'complex') #FIXME zerosが上手く動かない
        result = np.array(((0, 0, 0, 0, 0, 0),
                           (0, 0, 0, 0, 0, 0),
                           (0, 0, 0, 0, 0, 0),
                           (0, 0, 0, 0, 0, 0),
                           (0, 0, 0, 0, 0, 0),
                           (0, 0, 0, 0, 0, 0)), dtype = 'complex')
        # print(tomography[0][0])
        measurement_basis = basis.table_mb
        rho, t_rho, probability = self.two_tomography([])


        # probability = np.array(probability).reshape(6, 6)
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
                result[cols][lows] = probability[cols][lows] * tomography[cols][lows] #FIXME is this true?

        sum = result[0][0] + result[0][1] + result[1][0] + result[1][1]

        for cols in range(6):
            for lows in range(6):
                result[cols][lows] = result[cols][lows] / sum

        # print(result.reshape(6, 6))
        # print(result)
        # print("=====================")
        # print(sum)

        return self.tomograph_density(result)


    #
    @classmethod
    def tomograph_density(self, re):
        to_rho = 0
        state_list = self.state_list
        initial = self.initial

        for j in initial.values():
            st = []
            rho = 0

            S00 = re[0][0] + re[0][1] + re[1][0] + re[1][1]
            st.append(S00)
            # print(re[0][0])
            print(st)

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

            to_rho = to_rho + rho

        print("========== measure density matrix ==========")
        pprint.pprint(rho)

if __name__ == '__main__':
    # density_matrix.iterate_sim(0.3)#e_rate
    density_matrix.evaluation_density(1000)
     # density_matrix.tomography_simulater(0, 10000)
     # print("=======================")
     # density_matrix.two_tomography()
     # density_matrix.a
