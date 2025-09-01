import math
import numpy as np
import matplotlib.pyplot as plt
from prettytable import PrettyTable

# Sample data

Y = [31.086,28.393,23.914,20.946,17.623] 
X = [675,629,534,465,404]
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
    Σx2list.append(round(X[i]**2,4))
    Σy += Y[i]
    Σxy += X[i]*Y[i]
    Σxylist.append(round(X[i]*Y[i],4))
    Σx += X[i]
    count += 1
    Slno.append(count)

# display table(least square fitting)
table = PrettyTable()
table.field_names = ["Sl No.","X", "Y", "X^2", "XY"]
table.add_rows(list(zip(Slno, X, Y, Σx2list, Σxylist)))
table.add_row([" ",'','','',''])
table.add_row(["Total -", round(Σx,4), round(Σy,4), round(Σx2,4), round(Σxy,4)])
print(table)

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

#Printing slopes,intercepts and errors
table2 = PrettyTable()
table2.title = "Slope, Intercept and Errors"
table2.add_row(["Slope" , round(A,3) ])
table2.add_row(["Intercept" , round(B,3) ])
table2.add_row(["Error in y", round(error_y,3) ])
table2.add_row(["Delta" , round(delta,5) ])
table2.add_row(["Error in slope", round(error_slope,5) ])
table2.add_row(["Error in intercept", round(error_intercept,5) ])
print(table2)
print()

#Printing the function values
table3 = PrettyTable(["x" , "y = a_0 + a_1 x"])
table3.title ="Function values"
for i in X:
    table3.add_row([i, round(f(i),3)])
print(table3)
X_axis = 'B (in mT)'
Y_axis = '$\Delta k$'
#GRAPH PLOT
plt.xlabel(X_axis)
plt.ylabel(Y_axis)
plt.title(X_axis + " vs " + Y_axis)
plt.grid()
plt.scatter(X,Y,label='Data Points',s=20,c='orange')
fit = np.polyfit(X,Y,1)
x_fit = np.linspace(min(X),max(X),1000)
y_fit = fit[1]+fit[0]*x_fit
plt.plot(x_fit,y_fit,'--', label = f'Least-square fit line : y = ({round(A,3)}±{round(error_slope,3)})x + ({round(B,3)}±{round(error_intercept,3)})')
plt.legend()
plt.show()
