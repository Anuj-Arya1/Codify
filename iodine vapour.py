# Least squares fitting of observed wavelengths to a theoretical wavelength 
# import math
# import numpy as np
# import matplotlib.pyplot as plt
# from prettytable import PrettyTable

# # Sample data

 
# X = [5460,5790,5790,5960,6150,6200,6230,6910] # given lambda
# Y = [5460,5780,5810,5970,6140,6200,6250,6950] # observed lambda


# # TABLE DATA
# Σx2 = 0
# Σy = 0
# Σxy = 0
# Σx = 0
# count = 0
# Σxylist = []
# Σx2list = []
# Slno = [ ]

# for i in range(len(X)):
#     Σx2 += X[i]**2
#     Σx2list.append(round(X[i]**2,4))
#     Σy += Y[i]
#     Σxy += X[i]*Y[i]
#     Σxylist.append(round(X[i]*Y[i],4))
#     Σx += X[i]
#     count += 1
#     Slno.append(count)

# # display table(least square fitting)
# table = PrettyTable()
# table.field_names = ["Sl No.","X", "Y", "X^2", "XY"]
# table.add_rows(list(zip(Slno, X, Y, Σx2list, Σxylist)))
# table.add_row([" ",'','','',''])
# table.add_row(["Total -", round(Σx,4), round(Σy,4), round(Σx2,4), round(Σxy,4)])
# print(table)

# # Finding slope and intercept
# fit = np.polyfit(X,Y,1)
# A = fit[0]
# B = fit[1]
# def f(x):
#   return A*x + B
# # Error analysis
# error_sum = 0
# for i in range(len(X)):
#     error_sum += (Y[i] - f(X[i]))**2
# error_y = (error_sum/ (len(X) - 2))**0.5
# delta = len(X)*Σx2 - Σx**2
# error_slope = error_y * (len(X)/delta)**0.5
# error_intercept = error_y * (Σx2/delta)**0.5

# #Printing slopes,intercepts and errors
# table2 = PrettyTable()
# table2.title = "Slope, Intercept and Errors"
# table2.add_row(["Slope" , round(A,3) ])
# table2.add_row(["Intercept" , round(B,3) ])
# table2.add_row(["Error in y", round(error_y,3) ])
# table2.add_row(["Delta" , round(delta,5) ])
# table2.add_row(["Error in slope", round(error_slope,5) ])
# table2.add_row(["Error in intercept", round(error_intercept,5) ])
# print(table2)
# print()

# #Printing the function values
# table3 = PrettyTable(["x" , "y = a_0 + a_1 x"])
# table3.title ="Function values"
# for i in X:
#     table3.add_row([i, round(f(i),3)])
# print(table3)
# X_axis = r'$\lambda_{given}(\AA)$'
# Y_axis = r'$\lambda_{observed}$ $(\AA)$'
# #GRAPH PLOT
# plt.xlabel(X_axis)
# plt.ylabel(Y_axis)
# plt.title(X_axis + " vs " + Y_axis)
# plt.grid()
# plt.scatter(X,Y,label='Data Points',s=20,c='orange')
# fit = np.polyfit(X,Y,1)
# x_fit = np.linspace(min(X),max(X),1000)
# y_fit = fit[1]+fit[0]*x_fit
# plt.plot(x_fit,y_fit,'--', label = f'Least-square fit line : y = ({round(A,3)}±{round(error_slope,3)})x + ({round(B,3)}±{round(error_intercept,3)})')
# plt.legend()
# plt.show()
# L_obs = [6420,5800,5720,5300,5240,5150,5100,4800,4730,4680,4640,4580,4530,4510,4480,4420,4390,4330,4310,4290,4280,4060]
# L_obs = [6200,6100,5780,5690,5280,5210,5140,5090,4710,4680,4630,4570,4520,4500,4470,4270,4240]
# L_obs = [5640,5660,5690,5740,5760,5790,5830,5860,5910,5950,5990,6040,6080,6130,6160,6210,6250,6300,6340,6370] # iodine vapour
# L_corr = [round((l + 144.374) / 1.025, 3) for l in L_obs] # ROUNDED TO 3 DECIMAL PLACES
# print(L_corr)

# Iodine Vapour Absorption Spectrum Visualization
import matplotlib.pyplot as plt

nu_values = [17720.15, 17659.10, 17568.29, 17419.01, 17360.01, 17272.25, 17156.61, 17070.89, 16929.91, 16818.79, 16709.18, 16574.03, 16467.52, 16336.29, 16258.55, 16130.62, 16029.71, 15905.35, 15807.23, 15734.43]
nu_values.reverse()
plt.figure(figsize=(6, 8))
 
# Draw ground state (reference level)
plt.hlines(y=0, xmin=-1, xmax=1, color='black', linewidth=2)
plt.text(1.1, 0, r"$v''=0$ (ground state)", va='center')

# Draw excited state levels
for i, nu in enumerate(nu_values):
  plt.hlines(y=nu, xmin=-1, xmax=1, color='blue')
  plt.text(1.1, nu, f"v'={i}, {nu:.0f}", va='center', fontsize=8)

# Formatting
plt.ylabel("Energy / Wavenumber (cm$^{-1}$)")
plt.title("Iodine Absorption Spectrum: Energy Levels")
plt.ylim(15500, 18000)
plt.xticks([]) # remove x-axis ticks
plt.xlim(-1, 4)
plt.tight_layout()
plt.show()