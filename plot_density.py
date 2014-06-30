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
import matplotlib

#matplotlib.rc('text',usetex=True)

times,L,MLT,MLAT,InvLat,density=get_density_and_time('rbspa',datetime(2012,10,6),datetime(2012,10,10))

emfisis_fit=emfisis_fit_model('rbspa')

#fitdensity=[emfisis_fit(t,L,MLT,MLAT,InvLat) for t in times]

fitdensity,fituncert=emfisis_fit(times,L,MLT,MLAT,InvLat,returnUncert=True)
#fitdensity=emfisis_fit(times,L,0,0,1)

fig1=plt.figure()

fig1.gca().plot(times,density,linestyle='',marker='.')
fig1.gca().plot(times,fitdensity,color='g')
fig1.gca().plot(times,fitdensity+fituncert,linestyle=':',color='g')
fig1.gca().plot(times,fitdensity-fituncert,linestyle=':',color='g')

fig2=plt.figure()
fig2.gca().plot(L,density,linestyle='',marker='.')
fig2.gca().plot(L,fitdensity,color='g')
fig2.gca().plot(L,fitdensity+fituncert,linestyle=':',color='g')
fig2.gca().plot(L,fitdensity-fituncert,linestyle=':',color='g')

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

#fig3=plt.figure()

#plt.plot(times,ozhogin_density_latitude_factor(MLAT,InvLat))
#plt.xlabel('Universal time')
#plt.ylabel('Ozhogin latitude factor')

#fig3=plt.figure()

#plt.plot(times,MLT)

#plt.plot(times,sheeley_density_trough(L,MLT),label='Normal')
#plt.plot(times,124*(3.0/L)**4.0,label='Without MLT dependence')
#plt.plot(times,(36*(3.0/L)**3.5*np.cos((MLT-7.7*(3.0/L)**2.0+12)*np.pi/12))/sheeley_density_trough(L,MLT))
#plt.legend()
#plt.xlabel('Universal time')
#plt.ylabel('Sheeley trough density')

fh=open("fitcoeffs_new.csv",'w')
fh.write("Interval start (ordinal days), Interval end(ordinal days), Plasmapause L* (Re), Plasmapause width (Re), Plasmasphere multiplier, Trough multiplier")
np.savetxt(fh,emfisis_fit.fitcoeffs,delimiter=',')

Tlin,Llin=np.linspace(date2num(times[0]),date2num(times[-1]),100),np.linspace(1.5,6,100)
T,LL=np.meshgrid(Tlin,Llin)

fig4=plt.figure()
MLT=0
MLAT=0
InvLat=1
fitresult=emfisis_fit(T,LL,MLT,MLAT,InvLat).reshape(LL.shape)
cmap=plt.get_cmap('spectral')
im=plt.imshow(fitresult,origin='lower',extent=(Tlin.min(),Tlin.max(),Llin.min(),Llin.max()),aspect='auto',norm=matplotlib.colors.LogNorm(),cmap=cmap,clim=(1,30000))
#points=plt.scatter(times,L,c=density,edgecolors='none',cmap=cmap,norm=matplotlib.colors.LogNorm(),vmin=1,vmax=30000)

times,L,MLT,MLAT,InvLat,density=get_density_and_time('rbspb',datetime(2012,10,8),datetime(2012,10,9))
points=plt.scatter(times,L,c=density,edgecolors='none',cmap=cmap,norm=matplotlib.colors.LogNorm(),vmin=1,vmax=30000)
#plt.colorbar(points)
fig4.colorbar(im)

fig5=plt.figure()
fitdensity,fituncert=emfisis_fit(times,L,MLT,MLAT,InvLat,returnUncert=True)
fig5.gca().plot(times,density,linestyle='',marker='.')
fig5.gca().plot(times,fitdensity,color='g')
fig5.gca().plot(times,fitdensity+fituncert,linestyle=':',color='g')
fig5.gca().plot(times,fitdensity-fituncert,linestyle=':',color='g')
fig5.gca().set_yscale('log')

plt.show()
