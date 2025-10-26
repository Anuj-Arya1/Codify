
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

# Data from the experiment
f_r =[2.798,2.613,2.466,2.380,2.332,2.108,1.824,1.767,1.559] # in Hz
d =[5.4,5.9,6.4,6.9,7.9,8.9,11.9,12.9,16.9] #in cm 
B = [0.18,0.17,0.16,0.15,0.13,0.11,0.09,0.08,0.06] # in mT

# plot of f vs d
# d_1 =[ x**(-3/2) for x in d]

# def linear_fit(x, a, b):
#     for i in range(len(x)):
#       x[i] = float(x[i])
#     return a *x + b

# params, _ = curve_fit(linear_fit, d_1, f_r)
# f_linear_fit = linear_fit(d_1, params[0],params[1])

# plt.plot(d_1, f_linear_fit, color='blue', label='f = a*d^(-3/2) + b fit')
# plt.scatter(d_1, f_r, color='red', label='Data (f vs d)')
# plt.xlabel('d^(-3/2)')
# plt.ylabel('Frequency (Hz)')
# plt.title('Frequency vs d^(-3/2)')
# plt.legend()
# plt.grid()
# plt.show()
# print('Fit parameters (a, b):', params)


# Fit f as a function of sqrt(B)
def sqrt_fit(x, a, b):
    return a * np.sqrt(x) + b

params, _ = curve_fit(sqrt_fit, B, f_r)
f_sqrt_fit = sqrt_fit(B, *params)

plt.plot(B, f_sqrt_fit, color='blue', label='f = a*sqrt(B) + b fit')
plt.scatter(B, f_r, color='red', label='Data (f vs B)')
plt.xlabel('Magnetic field (mT)')
plt.ylabel('Frequency (Hz)')
plt.title('Frequency vs Magnetic field')
plt.legend()
plt.grid()
plt.show()
print('Fit parameters (a, b):', params)



# CALCULATIONS of TABLE
# q1 = [4.8,3.6,3.0,2.5,2.1,1.9,1.8,1.7,1.6]
# q2 = [5.6,4.4,3.5,3.0,2.7,2.3,2.0,1.8,1.7]
# q3 = [5.8,4.7,3.8,3.2,2.7,2.4,2.1,2.0,1.8]
# h = [0.49,0.65,0.77,0.93,1.07,1.19,1.33,1.46,1.58]
# h_new = [h*10 for h in h]
# p = 16
# import math as m

# Hpp = [round(2*m.sqrt(2)*i, 2) for i in h_new]
# q_p1 = [q1[i]/p for i in range(len(q1))]
# H_0 = [round(q_p1[i]*Hpp[i], 2) for i in range(len(q_p1))]

# print("H_pp =",Hpp)
# print("H_0 =",H_0)

# # q2
# Hpp2 = [round(2*m.sqrt(2)*i, 2) for i in h_new]
# q_p2 = [q2[i]/p for i in range(len(q2))]
# H_02 = [round(q_p2[i]*Hpp2[i], 2) for i in range(len(q_p2))]

# print("H_pp2 =",Hpp2)
# print("H_02 =",H_02)

# # q3
# Hpp3 = [round(2*m.sqrt(2)*i, 2) for i in h_new]
# q_p3 = [q3[i]/p for i in range(len(q3))]
# H_03 = [round(q_p3[i]*Hpp3[i], 2) for i in range(len(q_p3))]
# print("H_pp3 =",Hpp3)
# print("H_03 =",H_03)

# # g
# h = 6.625*10**-27 # erg s
# v1 = [12.8*10**6,14.83*10**6,15.81*10**6] # Hz
# u_B = 9.27*10**-21 # erg G^-1
# def g(h,v1,H_0,u_B):
#     g = (h*v1)/(H_0*u_B)
#     return round(g, 3)

# g1 = [g(h,v1[0],H_0[i],u_B) for i in range(len(H_0))]
# g2 = [g(h,v1[1],H_02[i],u_B) for i in range(len(H_02))]
# g3 = [g(h,v1[2],H_03[i],u_B) for i in range(len(H_03))]
# print("g1 =",g1)
# print("avg. g1 =",round(sum(g1)/len(g1), 3))
# print("g2 =",g2)
# print("avg. g2 =",round(sum(g2)/len(g2),3))
# print("g3 =",g3)
# print("avg. g3 =",round(sum(g3)/len(g3),3))
