import matplotlib.pyplot as plt
import math 
import numpy as np

def vsignal_calc(Vac,r1,r2):
    Vsig = []
    for i in range(len(Vac)):
        Vsig.append((round((Vac[i]/(2*math.sqrt(2)))*r2/(r1+r2) * 10**6,2)))
    return Vsig

def least_sq_fit_slope(X,Y,roundi):
    # TABLE DATA
    Σx2 = 0
    Σy = 0
    Σxy = 0
    Σx = 0
    count = 0
    Σxylist = []
    Σx2list = []
    Slno = [ ]

    for i in range(len(X)):
        Σx2 += X[i]**2
        Σx2list.append(round(X[i]**2,roundi))
        Σy += Y[i]
        Σxy += X[i]*Y[i]
        Σxylist.append(round(X[i]*Y[i],roundi))
        Σx += X[i]
        count += 1
        Slno.append(count)


    # Finding slope and intercept
    fit = np.polyfit(X,Y,1)
    A = fit[0]
    B = fit[1]
    def f(x):
        return A*x + B
    # Error analysis
    error_sum = 0
    for i in range(len(X)):
        error_sum += (Y[i] - f(X[i]))**2
    error_y = (error_sum/ (len(X) - 2))**0.5
    delta = len(X)*Σx2 - Σx**2
    error_slope = error_y * (len(X)/delta)**0.5
    # slope,error_slope

    slope = ["Slope" , round(A,roundi) ]
    return slope,error_slope



# freq - 600Hz
Vac1 = [1.005,1.535,2.150,2.50,3.00]
Vsig1 = vsignal_calc(Vac1, 4700, 12.8)  # mu V
Vdc1 = [0.096,0.186,0.293,0.354,0.437]
# freq - 900Hz
Vac2 = [1.015,1.405,2.01,2.45,3.00]
Vsig2 = vsignal_calc(Vac2, 4700, 12.8)  # mu V
Vdc2 = [0.098,0.164,0.266,0.349,0.434]
# freq - 1200Hz
Vac3 = [1.04,1.630,2.025,2.6,3.00]
Vsig3 = vsignal_calc(Vac3, 4700, 12.8)  # mu V
Vdc3 = [0.1,0.199,.268,0.372,0.432]
# freq - 1500Hz
Vac4 = [1.02,1.505,2.00,2.2,2.75,3.00]
Vsig4 = vsignal_calc(Vac4, 4700, 12.8)  # mu V
Vdc4 = [0.096,0.178,0.261,0.309,0.388,0.43]

S_50 = []
S_err_50 = []
s1,se1 = least_sq_fit_slope(Vsig1, Vdc1,8)
s2,se2 = least_sq_fit_slope(Vsig2, Vdc2,8)
s3,se3 = least_sq_fit_slope(Vsig3, Vdc3,8)
s4,se4 = least_sq_fit_slope(Vsig4, Vdc4,8)
for s in [s1, s2, s3, s4]:
    S_50.append(s[1])
for se in [se1, se2, se3, se4]:
    S_err_50.append(se)
print("Slopes at different frequencies for Gain 50:",S_50,"Slope Errors:",S_err_50)

# Gain 100
# freq - 600Hz
Vac5 = [1.005,1.535,2.150,2.50,3.00]
Vsig5 = vsignal_calc(Vac5, 4700, 12.8)
Vdc5 = [0.241,.414,0.666,0.744,0.899]
# freq - 900Hz
Vac6 = [1.005,1.5,2.04,2.45,2.95]
Vsig6 = vsignal_calc(Vac6, 4700, 12.8)
Vdc6 = [0.241,0.404,0.581,0.735,0.892]
# freq - 1200Hz
Vac7 = [1.04,1.530,2.015,2.4,3.00]
Vsig7 = vsignal_calc(Vac7, 4700, 12.8)
Vdc7 = [0.247,0.411,0.572,0.733,0.895]
# freq - 1500Hz
Vac8 = [1,1.515,2.04,2.2,2.75,3.00]
Vsig8 = vsignal_calc(Vac8, 4700, 12.8)
Vdc8 = [0.24,0.408,0.581,0.646,0.825,0.898]

S_100 = []
S_err_100 = []
s1,se1 = least_sq_fit_slope(Vsig5, Vdc5,8)
s2,se2 = least_sq_fit_slope(Vsig6, Vdc6,8)
s3,se3 = least_sq_fit_slope(Vsig7, Vdc7,8)
s4,se4 = least_sq_fit_slope(Vsig8, Vdc8,8)
for s in [s1, s2, s3, s4]:
    S_100.append(s[1])
for se in [se1, se2, se3, se4]:
    S_err_100.append(se)
print("Slopes at different frequencies for Gain 100:",S_100,"Slope Errors:",S_err_100)

# # gain 100
# plt.plot(Vsig5, Vdc5, 'o-', label='600 Hz')
# plt.plot(Vsig6, Vdc6, 's-', label='900 Hz')
# plt.plot(Vsig7, Vdc7, '^-', label='1200 Hz')
# plt.plot(Vsig8, Vdc8, 'd-', label='1500 Hz')
# plt.xlabel('Signal Voltage (uV)')
# plt.ylabel('DC Voltage (V)')
# plt.title('DC Voltage vs Signal Voltage at Different Frequencies (Gain 100)')
# plt.legend()
# plt.grid()
# plt.show()
# # gain 50

# plt.plot(Vsig1, Vdc1, 'o-', label='600 Hz')
# plt.plot(Vsig2, Vdc2, 's-', label='900 Hz')
# plt.plot(Vsig3, Vdc3, '^-', label='1200 Hz')
# plt.plot(Vsig4, Vdc4, 'd-', label='1500 Hz')
# plt.xlabel('Signal Voltage (uV)')
# plt.ylabel('DC Voltage (V)')
# plt.title('DC Voltage vs Signal Voltage at Different Frequencies (Gain 50)')
# plt.legend()
# plt.grid()
# plt.show()

