import numpy as np
H_state = [1, 0, 0, 1] #|0>
V_state = [1, 0, 0, -1] #|1>
D_state = [1, 1, 0, 0] #|+>
A_state = [1, -1, 0, 0] #|->
R_state = [1, 0, 1, 0] #|0>+i|1>/sqrt(2)
L_state = [1, 0, -1, 0] #|0>-i|1>/sqrt(2)

state_table = {"HH":[H_state,H_state], "HV":[H_state,V_state,1/6],"VH":[V_state,H_state,1/6], "VV":[V_state,V_state,1/3],
               "HD":[H_state,D_state], "HA":[H_state,A_state,1/4],"VD":[V_state,D_state,1/4], "VA":[V_state,A_state,1/4],
               "HR":[V_state,R_state], "HL":[H_state,L_state,1/4],"VR":[V_state,R_state,1/4], "VL":[V_state,L_state,1/4],
               "DH":[D_state,V_state], "DV":[D_state,V_state,1/4],"AH":[A_state,H_state,1/4], "AV":[A_state,V_state,1/4],
               "DD":[D_state,D_state], "DA":[D_state,A_state,1/6],"AD":[A_state,D_state,1/6], "AA":[A_state,A_state,1/3],
               "DR":[D_state,R_state], "DL":[D_state,L_state,1/4],"AR":[A_state,R_state,1/4], "AL":[A_state,L_state,1/4],
               "RH":[R_state,H_state,], "RV":[R_state,V_state,1/4],"LH":[L_state,H_state,1/4], "LV":[L_state,V_state,1/4],
               "RD":[R_state,D_state,1/4], "RA":[R_state,A_state,1/4],"LD":[L_state,D_state,1/4], "LA":[L_state,A_state,1/4],
               "RR":[R_state,R_state,1/6], "RL":[R_state,L_state,1/3],"LR":[L_state,R_state,1/3], "LL":[L_state,L_state,1/6]}

states = ["H", "V", "D", "A", "R", "L"]

sigma_0 = np.array(((1, 0), (0, 1)))
sigma_1 = np.array(((0, 1), (1, 0)))
sigma_2 = np.array(((0, 1j), (-1j, 0)))
sigma_3 = np.array(((1, 0), (0, -1)))
sigma_list = [sigma_0, sigma_1, sigma_2, sigma_3]
