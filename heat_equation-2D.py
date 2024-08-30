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

from PIL import Image

def make_gif(filenames, output_filename, duration=100):
  """Creates a GIF from a list of image filenames."""

  images = [Image.open(f) for f in filenames]
  images[0].save(output_filename, save_all=True, append_images=images[1:], optimize=True, duration=duration, loop=0)

def Gaussian2D(x,y,a=100.0,b_x=0.0,b_y=0.0,c_x=0.1,c_y=0.1):
  return a*math.exp(-((x-b_x)**2/c_x+(y-b_y)**2/c_y))

def Biscuits(x,y,oven_temperature=435.928,room_temperature=293,biscuit_size=0.1):
  d=0.9/3
  biscuit_locations=[[0,0],[0,d],[0,-d],[d,0],[-d,0],[d,d],[-d,-d],[d,-d],[-d,d]]
  for b in biscuit_locations:
    if (np.sqrt((x-b[0])**2+(y-b[1])**2)<=biscuit_size):
      return room_temperature
  return oven_temperature

"""
code up your own I.C. and see what happens
"""
def YourInitialCondition2D(x,y):
  return 1.0

def PlotHeatMap(u_t,xgrid,ygrid,t,dt,cmin,cmax,folder):
  plt.clf()
  plt.title(f"Temperature at t = {round(t*dt,5)} seconds")
  plt.xlabel("x (m)")
  plt.ylabel("y (m)")
  plt.pcolormesh(xgrid,ygrid,u_t, cmap=plt.cm.jet, vmin=cmin, vmax=cmax)
  plt.colorbar(label="heat (K)")
  plt.savefig(f"{folder}/{t}.png",dpi=300)
  return 0
  """
  The 2-D Heat Equation: du/dt = alpha * (du/dx + du/dy)
  
  heat: u(t,x,y) -- this is what we want to solve for.
  du/dt : change in heat over time
  du/dx : change in heat over space (in x)
  du/dy : change in heat over space (in y)
  alpha : heat diffusivity 
  """

  #https://www.agroengineering.org/index.php/jae/article/view/jae.2014.232
  #approximate biscuit thermal conductivity, for high baking time, varies from 0.15 to 0.19 Wm–1 K–1

# Constants
# Physical
alpha=0.17 #Watt/(meter*K)
oven_temperature=435.928 #K (325 F)
room_temperature=293 #K
# Grid
# Space
min_x=-0.5 #meter
max_x=0.5 #meter
min_y=-0.5 #meter
max_y=0.5 #meter
dh=0.01 #meter
dx=dh
dy=dh #meter
xgrid_size = int((max_x-min_x)/dx) + 1
ygrid_size = int((max_y-min_y)/dy) + 1
xgrid=[dx*i+min_x for i in range(xgrid_size)]
ygrid=[dy*i+min_y for i in range(ygrid_size)]
# Time
start_time = 0 #seconds
end_time = 1#25*60 #seconds
dt=dh**2/alpha*0.24 #seconds
print(f"dt: {dt}")
tgrid_size = int((end_time-start_time)/dt) + 1
tgrid=[dt*i+start_time for i in range(tgrid_size)]

#Courant–Friedrichs–Lewy condition
CFL = alpha*dt/dh**2
print (f"CFL: {CFL}")
if (CFL > 0.25):
  print("BAD CFL ****")
  exit()

# 2D heat Equation
#Explicit Scheme
#Foward 1st Order in Time
#Central 2nd Order in Space

gauss=True
if gauss:
  files=[]
  # Boundary conditions
  u_top = 0
  u_left = 0
  u_bottom = 0
  u_right = 0

  #Initial Condition
  heat=np.array([np.array([Gaussian2D(x,y) for y in ygrid]) for x in xgrid])
  # Set the boundary conditions
  heat[(xgrid_size-1):, :] = u_top
  heat[:, :1] = u_left
  heat[:1, 1:] = u_bottom
  heat[:, (xgrid_size-1):] = u_right
  PlotHeatMap(heat,xgrid,ygrid,0,dt,0,100,"gauss")
  files.append(f"gauss/{0}.png")
  gammax = alpha*dt/dx**2
  gammay = alpha*dt/dy**2
  next_heat=heat.copy()

  num_frames=250
  
  for t in range(1,num_frames):
    print(t)
    for i in range(1,xgrid_size-1):
      for j in range(1,ygrid_size-1):
        du_dx = gammax*(heat[i+1][j] -2*heat[i][j] + heat[i-1][j])
        du_dy = gammay*(heat[i][j+1] -2*heat[i][j] + heat[i][j-1])
        next_heat[i][j] = heat[i][j] + du_dy + du_dx
    heat=next_heat
    PlotHeatMap(heat,xgrid,ygrid,t,dt,0,100,"gauss")
    files.append(f"gauss/{t}.png")

  #animate
  make_gif(files, "gauss_heat2d.gif", duration=100)

biscuits=True
if biscuits:
  files=[]
  # Boundary conditions
  u_top = oven_temperature
  u_left = oven_temperature
  u_bottom = oven_temperature
  u_right = oven_temperature

  heat=np.array([np.array([Biscuits(x,y) for y in ygrid]) for x in xgrid])
  #Initial Condition
  # Set the boundary conditions
  heat[(xgrid_size-1):, :] = u_top
  heat[:, :1] = u_left
  heat[:1, 1:] = u_bottom
  heat[:, (xgrid_size-1):] = u_right
  PlotHeatMap(heat,xgrid,ygrid,0,dt,273,oven_temperature,"biscuit")
  files.append(f"biscuit/{0}.png")
  gammax = alpha*dt/dx**2
  gammay = alpha*dt/dy**2
  next_heat=heat.copy()

  num_frames=250
  
  for t in range(1,num_frames):
    print(t)
    for i in range(1,xgrid_size-1):
      for j in range(1,ygrid_size-1):
        du_dx = gammax*(heat[i+1][j] -2*heat[i][j] + heat[i-1][j])
        du_dy = gammay*(heat[i][j+1] -2*heat[i][j] + heat[i][j-1])
        next_heat[i][j] = heat[i][j] + du_dy + du_dx
    heat=next_heat
    PlotHeatMap(heat,xgrid,ygrid,t,dt,273,oven_temperature,"biscuit")
    files.append(f"biscuit/{t}.png")

  #animate
  make_gif(files, "biscuits_heat2d.gif", duration=100)