# Caliberation of LIA
import math 
import matplotlib.pyplot as plt
import math
import numpy as np
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


def least_sq_fit(X,Y):
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
        Σx2list.append(round(X[i]**2,8))
        Σy += Y[i]
        Σxy += X[i]*Y[i]
        Σxylist.append(round(X[i]*Y[i],8))
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
    error_intercept = error_y * (Σx2/delta)**0.5

    # slopes,intercepts and errors
    slope = ["Slope" , round(A,8) ]
    intercept = ["Intercept" , round(B,8) ]
    err_y = ["Error in y", round(error_y,8) ]
    delta = ["Delta" , round(delta,8) ]
    err_slope = ["Error in slope", round(error_slope,8) ]
    err_intercept = ["Error in intercept", round(error_intercept,8) ]
    return slope, err_slope,err_y,intercept,err_intercept, A, B, error_slope, error_intercept
# Caliberation data
def vsignal_calc(Vac,r1,r2):
    Vsig = []
    for i in range(len(Vac)):
        Vsig.append((round((Vac[i]/(2*math.sqrt(2)))*r2/(r1+r2) * 10**6,2)))
    return Vsig
# Gain = 50
# freq - 600Hz
Vac1 = [1.005,1.035,1.235,1.535,1.805,2.150,2.50,2.75,3.00]
Vsig1 = vsignal_calc(Vac1, 4700, 12.8)  # uV
Vdc1 = [0.096,0.1,0.132,0.186,0.232,0.293,0.354,0.395,0.437]
# freq - 900Hz
Vac2 = [0.83,1.015,1.15,1.405,1.705,2.01,2.2,2.45,2.7,2.9,3.00]
Vsig2 = vsignal_calc(Vac2, 4700, 12.8)  # uV
Vdc2 = [0.076,0.098,0.118,0.164,0.216,0.266,0.303,0.349,0.393,0.424,0.434]
# freq - 1200Hz
Vac3 = [1.04,1.105,1.415,1.630,2.025,2.3,2.6,3.00]
Vsig3 = vsignal_calc(Vac3, 4700, 12.8)  # uV
Vdc3 = [0.1,0.11,0.165,0.199,.268,0.315,0.372,0.432]
# freq - 1500Hz
Vac4 = [1.02,1.505,2.00,2.2,2.75,3.00]
Vsig4 = vsignal_calc(Vac4, 4700, 12.8)  # uV
Vdc4 = [0.096,0.178,0.261,0.309,0.388,0.43]

# Gain 100
# freq - 600Hz
Vac5 = [1.005,1.035,1.235,1.535,1.805,2.150,2.50,2.75,3.00]
Vsig5 = vsignal_calc(Vac5, 4700, 12.8)
Vdc5 = [0.241,0.248,0.312,.414,0.509,0.666,0.744,0.823,0.899]
# freq - 900Hz
Vac6 = [0.18,0.7,1.005,1.035,1.215,1.5,1.805,2.04,2.2,2.45,2.75,2.95]
Vsig6 = vsignal_calc(Vac6, 4700, 12.8)
Vdc6 = [0.015,0.163,0.241,0.249,0.306,0.404,0.507,0.581,0.646,0.735,0.822,0.892]
# freq - 1200Hz
Vac7 = [1.04,1.355,1.530,1.745,2.015,2.4,2.8,3.00]
Vsig7 = vsignal_calc(Vac7, 4700, 12.8)
Vdc7 = [0.247,0.352,0.411,0.488,0.572,0.733,0.825,0.895]
# freq - 1500Hz
Vac8 = [1,1.515,2.04,2.2,2.75,3.00]
Vsig8 = vsignal_calc(Vac8, 4700, 12.8)
Vdc8 = [0.24,0.408,0.581,0.646,0.825,0.898]

# print("Slopes at different frequencies for Gain 50:",S_50,"Slope Errors:",S_50_err)
# print("Freq: 600Hz",Vsig1)
# print("Freq: 900Hz",Vsig2)
# print("Freq: 1200Hz",Vsig3)
# print("Freq: 1500Hz",Vsig4)
# print("-----------------------")
# print("Freq: 600Hz Gain 100",Vsig5)
# print("Freq: 900Hz Gain 100",Vsig6) 
# print("Freq: 1200Hz Gain 100",Vsig7)
# print("Freq: 1500Hz Gain 100",Vsig8)

# PLots of Vdc vs Vsig 
# gain 50
# plt.plot(Vsig1, Vdc1, marker='o',linestyle='-', label='600Hz Gain 50', color='b') 
# plt.plot(Vsig2, Vdc2, marker='s',linestyle='-', label='900Hz Gain 50', color='g')
# plt.plot(Vsig3, Vdc3, marker='^',linestyle='-', label='1200Hz Gain 50', color='r')
# plt.plot(Vsig4, Vdc4, marker='d',linestyle='-', label='1500Hz Gain 50', color='c')
# plt.xlabel('$V_{sig}$ ($\mu$V)')
# plt.ylabel('$V_{dc}$ (V)')
# plt.legend()
# plt.grid()
# plt.show()

# # gain 100
# plt.plot(Vsig5, Vdc5, marker='o',linestyle='-', label='600Hz Gain 100', color='b') 
# plt.plot(Vsig6, Vdc6, marker='s',linestyle='-', label='900Hz Gain 100', color='g')
# plt.plot(Vsig7, Vdc7, marker='^', linestyle='-', label='1200Hz Gain 100', color='r')
# plt.plot(Vsig8, Vdc8, marker='d', linestyle='-', label='1500Hz Gain 100', color='c')
# plt.xlabel('$V_{sig}$ ($\mu$V)')
# plt.ylabel('$V_{dc}$ (V)')
# plt.legend()
# plt.grid()
# plt.show()


# mutual inductance
# Gain 100
# freq - 605Hz
Vac_m1 = [7,9,11,13,15]
Vac_rms_m1 = [v/(2*math.sqrt(2)) for v in Vac_m1]
Vdc_m1 = [0.027,0.046,0.065,0.09,0.116]

# freq - 900Hz
Vac_m2 = [7,9,11,13,15]
Vac_rms_m2 = [v/(2*math.sqrt(2)) for v in Vac_m2]
Vdc_m2 = [0.067,0.096,0.123,0.16,0.198]

# freq - 1200Hz
Vac_m3 = [7,9,11,13,15]
Vac_rms_m3 = [v/(2*math.sqrt(2)) for v in Vac_m3]
Vdc_m3 = [0.106,0.145,0.181,0.231,0.281]

# freq - 1500Hz
Vac_m4 = [7,9,11,13,15]
Vac_rms_m4 = [v/(2*math.sqrt(2)) for v in Vac_m4]
Vdc_m4 = [0.144,0.194,0.239,0.301,0.365]

# S_m =[]
# S_m_err = []
# s1_m, s1_m_err = least_sq_fit_slope(Vac_rms_m1, Vdc_m1,8)
# s2_m, s2_m_err = least_sq_fit_slope(Vac_rms_m2, Vdc_m2,8)
# s3_m, s3_m_err = least_sq_fit_slope(Vac_rms_m3, Vdc_m3,8)
# s4_m, s4_m_err = least_sq_fit_slope(Vac_rms_m4, Vdc_m4,8)
# for s in [s1_m, s2_m, s3_m, s4_m]:
#     S_m.append(s[1])
# print("Slopes at different frequencies for Mutual Inductance:",S_m)

# least squares fitting of Slope vs frequency
# X = [600,900,1200,1500] # Hz
# Y = S_m
# # print(least_sq_fit_slope(X,Y,8))
# slope_m, error_slope_m,_,intercept_m,error_intercept_m,A_m_m,B_m_m,error_slope_m,error_intercept_m = least_sq_fit(X,Y)
# #GRAPH PLOT
# plt.xlabel('Frequency (Hz)')
# plt.ylabel('Slope ')
# plt.title('Slope vs Frequency for Mutual Inductance Measurement at Gain 100')
# plt.grid()
# plt.scatter(X,Y,label='Data Points',s=20,c='r')
# fit = np.polyfit(X,Y,1)
# x_fit = np.linspace(min(X),max(X),1000) 
# y_fit = fit[1]+fit[0]*x_fit
# plt.plot(x_fit,y_fit,'--', label = f'Least-square fit line : y = ({round(A_m_m,8)}±{round(error_slope_m,8)})x + ({round(B_m_m,5)}±{round(error_intercept_m,5)})')
# plt.legend()
# plt.show()


# plots

# plt.plot(Vac_rms_m1, Vdc_m1, marker='o', linestyle='-', label='at freq. 605Hz ', color='b')
# plt.plot(Vac_rms_m2, Vdc_m2, marker='s', linestyle='-', label='at freq. 900Hz ', color='g')
# plt.plot(Vac_rms_m3, Vdc_m3, marker='^', linestyle='-', label='at freq. 1200Hz ', color='r')
# plt.plot(Vac_rms_m4, Vdc_m4, marker='d', linestyle='-', label='at freq. 1500Hz ', color='orange')
# plt.xlabel('$V_{ac}(rms)$ (V)')
# plt.ylabel('$V_{dc}$ (V)')
# plt.title('Mutual Inductance Calibration at Gain 100')
# plt.legend()
# plt.grid()
# plt.show()


# Low resistance measurement
# Gain 50

# freq - 300Hz
Vac_lr1 = [1.005,1.535,2.005,2.55,3.05]
Vacrms_lr1 = [v/(2*math.sqrt(2)) for v in Vac_lr1]
Vdc_lr1 = [0.059,0.126,0.186,0.252,0.310]

# freq - 600Hz
Vac_lr2 = [1.05,1.57,2.01,2.55,3.]
Vacrms_lr2 = [v/(2*math.sqrt(2)) for v in Vac_lr2]
Vdc_lr2 = [0.065,0.133,0.191,0.26,0.315]

# freq - 900Hz
Vac_lr3 = [1.005,1.16,2.03,2.5,3.00]
Vacrms_lr3 = [v/(2*math.sqrt(2)) for v in Vac_lr3]
Vdc_lr3 = [0.064,0.081,0.198,0.262,0.323]

# freq - 1205Hz
Vac_lr4 = [1.085,1.515,2.035,2.5,3.00]
Vacrms_lr4 = [v/(2*math.sqrt(2)) for v in Vac_lr4]
Vdc_lr4 = [0.071,0.125,0.197,0.260,0.323]

# freq - 1500Hz
Vac_lr5 = [1.06,1.5,2.025,2.6,3.00]
Vacrms_lr5 = [v/(2*math.sqrt(2)) for v in Vac_lr5]
Vdc_lr5 = [0.064,0.121,0.187,0.258,0.309]

# gain 100
# freq - 300Hz
Vac_lr11 = [1.005,1.505,2.008,2.5,3.05]
Vacrms_lr11 = [v/(2*math.sqrt(2)) for v in Vac_lr11]
Vdc_lr11 = [0.166,0.289,0.413,0.537,0.65]
# freq - 600Hz
Vac_lr22 = [1.01,1.515,2.01,2.5,3.]
Vacrms_lr22 = [v/(2*math.sqrt(2)) for v in Vac_lr22]
Vdc_lr22 = [0.173,0.301,0.427,0.544,0.667]
# freq - 900Hz
Vac_lr33 = [1.015,1.515,2.03,2.5,3.00]
Vacrms_lr33 = [v/(2*math.sqrt(2)) for v in Vac_lr33]
Vdc_lr33 = [0.064,0.13,0.199,0.265,0.327]
# freq - 1205Hz
Vac_lr44 = [1.065,1.525,2.015,2.5,3.00]
Vacrms_lr44 = [v/(2*math.sqrt(2)) for v in Vac_lr44]
Vdc_lr44 = [0.187,0.305,0.436,0.566,0.690]
# freq - 1500Hz
Vac_lr55 = [1.03,1.535,2.015,2.55,3.00]
Vacrms_lr55 = [v/(2*math.sqrt(2)) for v in Vac_lr55]
Vdc_lr55 = [0.172,0.297,0.417,0.548,0.656]



# plots
# gain 100
# plt.plot(Vacrms_lr11, Vdc_lr11, marker='o', linestyle='-', label='at freq. 300Hz ', color='b')
# plt.plot(Vacrms_lr22, Vdc_lr22, marker='s', linestyle='-', label='at freq. 600Hz ', color='g')
# # plt.plot(Vacrms_lr33, Vdc_lr33, marker='^', linestyle='-', label='at freq. 900Hz ', color='r')
# plt.plot(Vacrms_lr44, Vdc_lr44, marker='d', linestyle='-', label='at freq. 1200Hz ', color='c')
# plt.plot(Vacrms_lr55, Vdc_lr55, marker='x', linestyle='-', label='at freq. 1500Hz ', color='m')
# plt.xlabel('$V_{ac(rms)}$ (V)')
# plt.ylabel('$V_{dc}$ (V)')
# plt.title('Low Resistance Measurement at Gain 100')     
# plt.legend()
# plt.grid()
# plt.show()

# # plots 
# # gain 50
# plt.plot(Vacrms_lr1, Vdc_lr1, marker='o', linestyle='-', label='at freq. 300Hz ', color='b')
# plt.plot(Vacrms_lr2, Vdc_lr2, marker='s', linestyle='-', label='at freq. 600Hz ', color='g')
# plt.plot(Vacrms_lr3, Vdc_lr3, marker='^', linestyle='-', label='at freq. 900Hz ', color='r')    
# plt.plot(Vacrms_lr4, Vdc_lr4, marker='d', linestyle='-', label='at freq. 1200Hz ', color='c')
# plt.plot(Vacrms_lr5, Vdc_lr5, marker='x', linestyle='-', label='at freq. 1500Hz ', color='m')
# plt.xlabel('$V_{ac(rms)}$ (V)') 
# plt.ylabel('$V_{dc}$ (V)')
# plt.title('Low Resistance Measurement at Gain 50')
# plt.legend()
# plt.grid()
# plt.show()

# least squares fitting
SL_50 =[]
SL_50_err = []
s1_50,s1_50_err = least_sq_fit_slope(Vacrms_lr1, Vdc_lr1,8)
s2_50,s2_50_err = least_sq_fit_slope(Vacrms_lr2, Vdc_lr2,8)
s3_50,s3_50_err = least_sq_fit_slope(Vacrms_lr3, Vdc_lr3,8)
s4_50,s4_50_err = least_sq_fit_slope(Vacrms_lr4, Vdc_lr4,8)
s5_50,s5_50_err = least_sq_fit_slope(Vacrms_lr5, Vdc_lr5,8)
SL_50.append(s1_50)
SL_50.append(s2_50)
SL_50.append(s3_50)
SL_50.append(s4_50)
SL_50.append(s5_50)
SL_50_err.append(s1_50_err)
SL_50_err.append(s2_50_err)
SL_50_err.append(s3_50_err)
SL_50_err.append(s4_50_err)
SL_50_err.append(s5_50_err)



print("Slopes at different frequencies for Low Resistance Measurement Gain 50:",SL_50)
print("Avg of error in slopes", np.mean(SL_50_err))

#gain 100
SL_100 =[]
SL_100_err = []
s1_100,s1_100_err = least_sq_fit_slope(Vacrms_lr11, Vdc_lr11,8)
s2_100,s2_100_err = least_sq_fit_slope(Vacrms_lr22, Vdc_lr22,8)
s3_100,s3_100_err = least_sq_fit_slope(Vacrms_lr33, Vdc_lr33,8)
s4_100,s4_100_err = least_sq_fit_slope(Vacrms_lr44, Vdc_lr44,8)
s5_100,s5_100_err = least_sq_fit_slope(Vacrms_lr55, Vdc_lr55,8)
SL_100.append(s1_100)
SL_100.append(s2_100)
SL_100.append(s3_100)
SL_100.append(s4_100)
SL_100.append(s5_100)
SL_100_err.append(s1_100_err)
SL_100_err.append(s2_100_err)
SL_100_err.append(s3_100_err)
SL_100_err.append(s4_100_err)
SL_100_err.append(s5_100_err)

print("Avg of error in slopes", np.mean(SL_100_err))
print("Slopes at different frequencies for Low Resistance Measurement Gain 100:",SL_100)
