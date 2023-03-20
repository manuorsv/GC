#!/usr/bin/env python
# coding: utf-8

# -*- coding: utf-8 -*-
"""
Created on Tue Mar 10 18:58:33 2020

@author: Robert
"""

#from mpl_toolkits import mplot3d

import os
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import axes3d

vuestra_ruta = ""

os.getcwd()
#os.chdir(vuestra_ruta)


#APARTADO 1
"""
2-sphere
"""


u = np.linspace(0, np.pi, 30)
v = np.linspace(0, 2 * np.pi, 60)

x = np.outer(np.sin(u), np.sin(v))
y = np.outer(np.sin(u), np.cos(v))
z = np.outer(np.cos(u), np.ones_like(v))

p = 6
q = 5
t2 = np.linspace(0, 2*np.pi, 360)
x2 = np.sin(p*t2)*np.cos(q*t2)
y2 = np.sin(p*t2)*np.sin(q*t2)
z2 = np.cos(p*t2)

fig = plt.figure()
ax = plt.axes(projection='3d')
#ax.plot(x2, y2, z2, '-b',c="gray")

ax.plot_surface(x, y, z, rstride=1, cstride=1,
                cmap='viridis', edgecolor='none')

ax.plot(x2, y2, z2, '-b',c="gray",zorder=3)
#ax.plot_wireframe(x2, y2, z2)

ax.set_title('surface');

def proj(x,z,z0=1,alpha=1):
    z0 = z*0+z0
    eps = 1e-16
    x_trans = x/(abs(z0-z)**alpha+eps)
    return(x_trans)
    #Nótese que añadimos un épsilon para evitar dividir entre 0!!

z0 = 1

fig = plt.figure(figsize=(12,12))
fig.subplots_adjust(hspace=0.4, wspace=0.2)

c2 = np.sqrt(x2**2+y2**2)
col = plt.get_cmap("hot")(c2/np.max(c2))

ax = fig.add_subplot(2, 2, 1, projection='3d')
ax.plot(x2, y2, z2, '-b',c='gray',zorder=3)
ax.plot_surface(x, y, z, rstride=1, cstride=1, cmap='viridis', edgecolor='none')
#ax.scatter(x2, y2, z2, '-b',c=col,zorder= 3,s=3)
ax.set_title('2-sphere');
#ax.text(0.5, 90, 'PCA-'+str(i), fontsize=18, ha='center')

ax = fig.add_subplot(2, 2, 2, projection='3d')
ax.set_xlim3d(-1,1)
ax.set_ylim3d(-1,1)
#ax.set_zlim3d(0,1000)
ax.plot_surface(proj(x,z,z0=z0,alpha=1/2), proj(y,z,z0=z0,alpha=1/2), z*0+1, rstride=1, cstride=1, cmap='viridis', alpha=0.5, edgecolor='purple')

#ax.plot(proj(x2,z2,z0=z0,alpha=1/2), proj(y2,z2,z0=z0,alpha=1/2), 1,'-b',c='gray',zorder=3)
ax.scatter(proj(x2,z2,z0=z0,alpha=1/2), proj(y2,z2,z0=z0,alpha=1/2), 1, '-b',c=col,zorder= 3,s=10)
ax.set_title('Stereographic projection');

plt.show()
fig.savefig('stereo2.png', dpi=250)   # save the figure to file
plt.close(fig) 


"""
APARTADO 2
"""

from matplotlib import animation
#from mpl_toolkits.mplot3d.axes3d import Axes3D

def proj2(x,z,t,z0=1,alpha=1):
    z0 = z*0+z0
    eps = 1e-16
    x_trans = (2*x)/(2*(1-t)+t*(z0-z)+eps)
    return(x_trans)
    #Nótese que añadimos un épsilon para evitar dividir entre 0!!

def animate(t):
    xt = proj2(x,z,t,z0)
    yt = proj2(y,z,t,z0)
    zt = -t + z*(1-t)
    x2t = proj2(x2,z2,t,z0)
    y2t = proj2(y2,z2,t,z0)
    z2t = -t + z2*(1-t)
    
    ax = plt.axes(projection='3d')
    ax.set_zlim3d(-1,1)
    ax.plot_surface(xt, yt, zt, rstride=1, cstride=1, alpha=0.5,
                    cmap='viridis', edgecolor='none')
    ax.plot(x2t,y2t, z2t, '-b',c="gray")
    return ax,

def init():
    return animate(0),

animate(np.arange(0, 1,0.1)[1])
plt.show()


fig = plt.figure(figsize=(6,6))
ani = animation.FuncAnimation(fig, animate, np.arange(0,1,0.05), init_func=init,
                              interval=20)
#ani.save("ejemplo.htm", fps = 5) 
ani.save("anim.gif", fps = 5)

"""
Alternativas para hacer animación
#celluloid
pypi.org/project/celluloid

#APNG
from apng import APNG
APNG.from_files(['atleta-01.jpg',
                 'atleta-02.jpg', 
                 'atleta-03.jpg',
                 'atleta-04.jpg',
                 'atleta-05.jpg'], 
                 delay=100).save('animatleta1.png')
"""



