import emfisis
import ephemeris
from matplotlib import pyplot as plt
from datetime import datetime, timedelta
from matplotlib.dates import date2num, num2date
import numpy as np
from smooth import smooth
from scipy.optimize import leastsq
import matplotlib
from densitymodels import *

#matplotlib.rc('text',usetex=True)

times,L,MLT,MLAT,InvLat,density=get_density_and_time('rbspa',datetime(2012,10,6),datetime(2012,10,9))

emfisis_fit=emfisis_fit_model('rbspa')

#fitdensity=[emfisis_fit(t,L,MLT,MLAT,InvLat) for t in times]

fitdensity=emfisis_fit(times,L,MLT,MLAT,InvLat)

fig1=plt.figure()

fig1.gca().plot(times,density,linestyle='',marker='.')
fig1.gca().plot(times,fitdensity)

fig2=plt.figure()
fig2.gca().plot(L,density,linestyle='',marker='.')
fig2.gca().plot(L,fitdensity)

fig2.gca().set_yscale('log')
fig2.gca().set_ylim(1e0,1e4)
fig2.gca().set_ylabel(r'Electron density (cm^-3)')
fig2.gca().set_xlabel('L shell')

#L=np.linspace(2,5.5)
#plt.plot(L,fitfunc(L,0,3.6,0.8,1,1))

fig1.gca().set_yscale('log')
fig1.gca().set_ylim(1e0,1e4)
fig1.gca().set_ylabel(r'Electron density (cm^-3)')
fig1.gca().set_xlabel('Universal time')

fig3=plt.figure()

plt.plot(times,ozhogin_density_latitude_factor(MLAT,InvLat))
plt.xlabel('Universal time')
plt.ylabel('Ozhogin latitude factor')

fig3=plt.figure()

#plt.plot(times,MLT)

#plt.plot(times,sheeley_density_trough(L,MLT),label='Normal')
#plt.plot(times,124*(3.0/L)**4.0,label='Without MLT dependence')
plt.plot(times,(36*(3.0/L)**3.5*np.cos((MLT-7.7*(3.0/L)**2.0+12)*np.pi/12))/sheeley_density_trough(L,MLT))
plt.legend()
plt.xlabel('Universal time')
plt.ylabel('Sheeley trough density')

fh=open("fitcoeffs_new.csv",'w')
fh.write("Interval start (ordinal days), Interval end(ordinal days), Plasmapause L* (Re), Plasmapause width (Re), Plasmasphere multiplier, Trough multiplier")
np.savetxt(fh,emfisis_fit.fitcoeffs,delimiter=',')

#T,LL=np.meshgrid(times,L)

#fig4=plt.figure()
#plt.contourf(T,LL,emfisis_fit(T,LL,MLT,MLAT,InvLat))

plt.show()
