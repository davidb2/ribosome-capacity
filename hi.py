import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_theme()

#Create radius and theta arrays, and a 2d radius/theta array
radius = np.linspace(0.2,0.4,51)
theta = np.linspace(0,2*np.pi,51)
R,T  = np.meshgrid(radius,theta)

#Calculate some values to plot
Zfun = lambda R,T: R**2*np.cos(T)
Z = Zfun(R,T)

#Create figure and polar axis
fig = plt.figure()
ax = fig.add_subplot(111, polar = True)

ax.pcolor(T,R,Z)    #Plot calculated values

#Plot thick red section and label it
theta = np.linspace(0,np.pi/4,21)
ax.plot(theta,[1.23 for t in theta],color='#AA5555',linewidth=10)   #Colors are set by hex codes
ax.text(np.pi/8,1.25,"Text")

ax.set_rmax(1.25)   #Set maximum radius

#Turn off polar labels
ax.axes.get_xaxis().set_visible(False)
ax.axes.get_yaxis().set_visible(False)
plt.show()
