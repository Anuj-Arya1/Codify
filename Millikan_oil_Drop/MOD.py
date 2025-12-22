
import numpy as np
# Milikan oil drop experiment
# Dynamic method data storage
# fall time (s)
f1 = [8.9,8.5,9.1,9.4,9.8] # drop 1
f2 = [12.5,13.9,13.7,14.4] # drop 2
f3 = [4.9,4.9,5.0,4.9,4.9,5.1] # drop 3
f4 = [6.5,6.4,6.4,6.0,6.2,6.4,6.4,6.0] # drop 4
f5 = [9.8,10.4,10.1,10.6,9.6,9.8] # drop 5

# rise time (s)
r1 = [6.3,6.4,5.9,6.0,5.9] # drop 1
r2 = [3.5,3.6,3.4,3.6] # drop 2
r3 = [4.6,4.4,4.5,4.4,4.5,4.8] # drop 3
r4 = [3.1,3.1,3.2,3.3,3.0,3.1,3.4,3.2] # drop 4
r5 = [3.8,3.8,3.8,3.5,3.6,3.8] # drop 5

# voltage (V)
Ve = [321,245,377,211,264] # drop 1-5 (in Volts)

def mean_(data):
    return round(sum(data) / len(data),2)

# rise time and fall time mean values
f_mean = [mean_(f1), mean_(f2), mean_(f3), mean_(f4), mean_(f5)]
r_mean = [mean_(r1), mean_(r2), mean_(r3), mean_(r4), mean_(r5)]
# print("Mean Fall Times(dynamic method):", f_mean)
# print("Mean Rise Times(dynamic method):", r_mean)

# mean free fall velocity
v_f = [round((0.002 / t)*10**4,2) for t in f_mean]  # assuming distance = 2 mm = 0.002 m
# print("Mean Free Fall Velocities(dynamic method) (m/s):", v_f)

# Balancing method data storage
# fall time (s)
tf1 = [10.3,10.8,10.8,10.6,10.7,11.2] # drop 1
tf2 = [2.2,2.2,2.2,2.1,2.4,2.2] # drop 2
tf3 = [4.3,4.3,4.2,4.7,4.4] # drop 3
tf4 = [3.6,3.6,3.6,3.5,3.4,3.5] # drop 4
tf5 = [2.5,2.5,2.6,2.5,2.5,2.4] # drop 5

# voltage (V)
vb1 = [145,160,146,141,160,157] # drop 1
vb2 = [208,214,210,208,205,203] # drop 2
vb3 = [274,274,277,273,279] # drop 3
vb4 = [141,141,141,143,141,141] # drop 4
vb5 = [206,204,205,212,208,215] # drop 5
# mean voltage (use scalar mean for each set)
vb_m1 = sum(vb1) / len(vb1)
vb_m2 = sum(vb2) / len(vb2)
vb_m3 = sum(vb3) / len(vb3)
vb_m4 = sum(vb4) / len(vb4)
vb_m5 = sum(vb5) / len(vb5)


# mean fall time for balancing method
tf_mean = [mean_(tf1), mean_(tf2), mean_(tf3), mean_(tf4), mean_(tf5)]
# print("Mean Fall Times (Balancing Method):", tf_mean)

# mean velocity for balancing method
v_bal = [round((0.002 / t)*10**4,2) for t in tf_mean]  # assuming distance = 2 mm = 0.002 m
# print("Mean Velocities (Balancing Method) (m/s):", v_bal)

# calculations 
# dynamic method
pho = 929 # density of oil (kg/m^3)
pho_a = 1 # density of air (kg/m^3)
g = 9.81 # acceleration due to gravity (m/s^2)
eta = 1.8432 * 10**-5 # viscosity of air (Pa.s)
d = 5e-3 # distance between plates (m)
T = 27.8 # temperature in degree Celsius
P = 0.761 # atmospheric pressure in m Hg
c = 6.17e-8 # m of Hg-m
C = 190.13
D = 9.199e-9  
zeta = 4.06e-8 # correction factor
# eta = 1.8610e-5 # viscosity of air at 27.8 degree Celsius (Pa.s)
xi = [D*v*1e-4 for v in v_f]  
r = [(-zeta + (zeta**2 + x)**0.5) for x in xi]
r3 = [ri**3 for ri in r]    
T_d = [(1+ f_mean[i]/r_mean[i]) for i in range(len(f_mean))]
ne = [(C*T_d[i]*r3[i])/Ve[i] for i in range(len(r3))]

print("Xi (dynamic method):", xi,'\n')
print("Radius (m) (dynamic method):", r,'\n')
print("Radius cubed (m^3) (dynamic method):", r3,'\n')
print("T_d (dynamic method):", T_d,'\n')
print("Calculated charge (C) (dynamic method):", ne,'\n')

# balancing method

xi_b = [D*v*1e-4 for v in v_bal]
r_b = [(-zeta + (zeta**2 + x)**0.5) for x in xi_b]
r3_b = [ri**3 for ri in r_b]
ne_b = []
# use the scalar mean voltages and append to ne_b (balancing-method results)
ne_b.append((C*r3_b[0]) / vb_m1)
ne_b.append((C*r3_b[1]) / vb_m2)
ne_b.append((C*r3_b[2]) / vb_m3)
ne_b.append((C*r3_b[3]) / vb_m4)
ne_b.append((C*r3_b[4]) / vb_m5)

print("Xi (balancing method):", xi_b,'\n')
print("Radius (m) (balancing method):", r_b,'\n')
print("Radius cubed (m^3) (balancing method):", r3_b,'\n')
print("Calculated charge (C) (balancing method):", ne_b,'\n')


   

