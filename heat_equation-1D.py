"""
Abraham R. Flores : 8/29/2024
QM 217: Introduction to Computational Partial Differential Equations
heat_equation.py: compute and visualize the 1D and 2D heat equation
"""

import matplotlib.pyplot as plt
import math
import numpy as np
from scipy.sparse import diags
import matplotlib.animation as animation

#1D gaussian
def Gaussian1D(x,a=1.0,b=0.0,c=0.01):
  return a*math.exp(-(x-b)**2/c)

"""
code up your own I.C. and see what happens
"""
def YourInitialCondition1D(x):
  return 1.0


"""
The 1-D Heat Equation: du/dt = alpha * du/dx

heat: u(t,x) -- this is what we want to solve for.
du/dt : change in heat over time
du/dx : change in heat over space
alpa h: heat diffusivity 
"""

# Constants
# Physical
alpha=0.17 #Watt/(meter*K)
oven_temperature=435.928 #K (325 F)
room_temperature=293 #K
# Grid
# Space
min_x=-0.5 #meter
max_x=0.5 #meter
dx=0.01 #meter
xgrid_size = int((max_x-min_x)/dx) + 1
xgrid=[dx*i+min_x for i in range(xgrid_size)]
# Time
start_time = 0 #seconds
end_time = 10#25*60 #seconds
dt=dx**2/alpha*0.49 #seconds
print(f"dt: {dt}")
tgrid_size = int((end_time-start_time)/dt) + 1
tgrid=[dt*i+start_time for i in range(tgrid_size)]

#Courant–Friedrichs–Lewy condition
CFL = alpha*dt/dx**2
print (f"CFL: {CFL}")
if (CFL > 0.5):
  print("BAD CFL ****")
  exit()

# 1D heat Equation
#Explicit Scheme
#Foward 1st Order in Time
#Central 2nd Order in Space
#Boundary Conditions: u(t,min_x)=0 , u(t,max_x)=0
#Initial Conditions: u(0,x) = gaussian(x)

gamma = alpha*dt/dx**2
k = [gamma*np.ones(xgrid_size-1),(1-2*gamma)*np.ones(xgrid_size),gamma*np.ones(xgrid_size-1)]
offset = [-1,0,1]
FTCS = diags(k,offset).toarray()
FTCS[0]=[0 for x in xgrid]
FTCS[-1]=[0 for x in xgrid]
heat=np.array([Gaussian1D(x) for x in xgrid])
data=[heat]

for i in range(1,tgrid_size):
  data.append(np.matmul(FTCS,data[i-1]))

def animate(i):
  line.set_ydata(data[i])  # update the data
  return line,

def init():
  line.set_ydata(np.ma.array(xgrid, mask=True))
  return line,

fig, ax = plt.subplots()
line, = ax.plot(np.array(xgrid), data[0], lw=2)
ax.set(xlim=[min_x,max_x], ylim=[0,1], xlabel='X [m]', ylabel='heat [?]')
ani = animation.FuncAnimation(fig, animate, frames=1000, init_func=init,interval=15, blit=True)
ani.save('1dHeatEq.gif', writer='imagemagick')