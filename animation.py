import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import ast
from main import *
import numpy as np
from matplotlib import animation
from plots import *

f = open("2.txt", "r+")
x2 = ast.literal_eval(f.readline())
y2 = ast.literal_eval(f.readline())
z2 = ast.literal_eval(f.readline())
f.close()

f = open("3.txt", "r+")
x3 = ast.literal_eval(f.readline())
y3 = ast.literal_eval(f.readline())
z3 = ast.literal_eval(f.readline())
f.close()

f = open("4.txt", "r+")
x4 = ast.literal_eval(f.readline())
y4 = ast.literal_eval(f.readline())
z4 = ast.literal_eval(f.readline())
f.close()

f = open("5.txt", "r+")
x5 = ast.literal_eval(f.readline())
y5 = ast.literal_eval(f.readline())
z5 = ast.literal_eval(f.readline())
f.close()

f = open("6.txt", "r+")
x6 = ast.literal_eval(f.readline())
y6 = ast.literal_eval(f.readline())
z6 = ast.literal_eval(f.readline())
f.close()

'''f = open("7.txt", "r+")
x7 = ast.literal_eval(f.readline())
y7 = ast.literal_eval(f.readline())
z7 = ast.literal_eval(f.readline())
f.close()

f = open("8.txt", "r+")
x8 = ast.literal_eval(f.readline())
y8 = ast.literal_eval(f.readline())
z8 = ast.literal_eval(f.readline())
f.close()

f = open("9.txt", "r+")
x9 = ast.literal_eval(f.readline())
y9 = ast.literal_eval(f.readline())
z9 = ast.literal_eval(f.readline())
f.close()

f = open("10.txt", "r+")
x10 = ast.literal_eval(f.readline())
y10 = ast.literal_eval(f.readline())
z10 = ast.literal_eval(f.readline())
f.close()
'''


t = time

dataSet2 = np.array([x2, y2, z2])
dataSet3 = np.array([x3, y3, z3])
dataSet4 = np.array([x4, y4, z4])
dataSet5 = np.array([x5, y5, z5])
dataSet6 = np.array([x6, y6, z6])
'''dataSet7 = np.array([x7, y7, z7])
dataSet8 = np.array([x8, y8, z8])
dataSet9 = np.array([x9, y9, z9])
dataSet10 = np.array([x10, y10, z10])'''
numDataPoints = len(t)

def animate_func(num):
   ax.clear()
   ax.plot3D(dataSet2[0, :num+1], dataSet2[1, :num+1], dataSet2[2, :num+1], c='red')
   ax.plot3D(dataSet3[0, :num + 1], dataSet3[1, :num + 1], dataSet3[2, :num + 1], c='green')
   ax.plot3D(dataSet4[0, :num + 1], dataSet4[1, :num + 1], dataSet4[2, :num + 1], c='purple')
   ax.plot3D(dataSet5[0, :num + 1], dataSet5[1, :num + 1], dataSet5[2, :num + 1], c='orange')
   ax.plot3D(dataSet6[0, :num + 1], dataSet6[1, :num + 1], dataSet6[2, :num + 1], c='blue')
   '''ax.plot3D(dataSet7[0, :num + 1], dataSet7[1, :num + 1], dataSet7[2, :num + 1], c='yellow')
   ax.plot3D(dataSet8[0, :num + 1], dataSet8[1, :num + 1], dataSet8[2, :num + 1], c='brown')
   ax.plot3D(dataSet9[0, :num + 1], dataSet9[1, :num + 1], dataSet9[2, :num + 1], c='pink')
   ax.plot3D(dataSet10[0, :num + 1], dataSet10[1, :num + 1], dataSet10[2, :num + 1], c='grey')'''

   ax.scatter(dataSet2[0, num], dataSet2[1, num], dataSet2[2, num], c='red', marker='o')
   ax.scatter(dataSet3[0, num], dataSet3[1, num], dataSet3[2, num], c='green', marker='o')
   ax.scatter(dataSet4[0, num], dataSet4[1, num], dataSet4[2, num], c='purple', marker='o')
   ax.scatter(dataSet5[0, num], dataSet5[1, num], dataSet5[2, num], c='orange', marker='o')
   ax.scatter(dataSet6[0, num], dataSet6[1, num], dataSet6[2, num], c='blue', marker='o')
   '''ax.scatter(dataSet7[0, num], dataSet6[1, num], dataSet7[2, num], c='yellow', marker='o')
   ax.scatter(dataSet8[0, num], dataSet6[1, num], dataSet8[2, num], c='brown', marker='o')
   ax.scatter(dataSet9[0, num], dataSet6[1, num], dataSet9[2, num], c='pink', marker='o')
   ax.scatter(dataSet10[0, num], dataSet6[1, num], dataSet10[2, num], c='grey', marker='o')'''


   ax.set_xlim3d([-2, 2])
   ax.set_ylim3d([-2, 2])
   ax.set_zlim3d([-2, 2])

   ax.set_title('Trajectory \nTime = ' + str(np.round(t[num], decimals=2)) + ' sec')
   ax.set_xlabel('x')
   ax.set_ylabel('y')
   ax.set_zlabel('z')


fig = plt.figure()
ax = plt.axes(projection='3d')
line_ani1 = animation.FuncAnimation(fig, animate_func, interval=0.0001, frames=numDataPoints)
plt.show()
