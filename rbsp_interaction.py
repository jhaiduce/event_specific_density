import emfisis
import ephemeris
import matplotlib
matplotlib.use('GTKAgg') # do this before importing pylab
from matplotlib import pyplot as plt
from datetime import datetime, timedelta
from matplotlib.dates import date2num, num2date
import numpy as np
from smooth import smooth
from scipy.optimize import leastsq
import matplotlib
from densitymodels import *
import matplotlib.dates as mdates
from matplotlib.widgets import Slider

times,L,MLT,MLAT,InvLat,density=get_density_and_time('rbspa',datetime(2012,10,6),datetime(2012,10,10))
otimes=date2num(times)

timesb,Lb,MLTb,MLATb,InvLatb,densityb=get_density_and_time('rbspb',datetime(2012,10,8),datetime(2012,10,9))
otimesb=date2num(timesb)

emfisis_fit=emfisis_fit_model('rbspa')

fig=plt.figure()
pax=plt.subplot(2,2,1,polar=True)
tax=plt.subplot(2,1,2)
fitax=plt.subplot(2,2,2)
cmap=plt.get_cmap('spectral')
Llin=np.linspace(1.5,6,100)
MLTlin=np.linspace(0,24,100)
MLTg,LL=np.meshgrid(MLTlin,Llin)
Tlin=np.linspace(date2num(times[0]),date2num(times[-1]),100)
T,LL=np.meshgrid(Tlin,Llin)

def initialize_tax():
    tax.clear()
    step=10
    fitresult=emfisis_fit(T,LL,MLT=0,MLAT=0,InvLat=1).reshape(LL.shape)
    im=tax.imshow(fitresult,origin='lower',extent=(Tlin.min(),Tlin.max(),Llin.min(),Llin.max()),aspect='auto',norm=matplotlib.colors.LogNorm(),clim=(1,30000))
    points=tax.scatter(timesb[::step],Lb[::step],c=densityb[::step],edgecolors='none',cmap=cmap,norm=matplotlib.colors.LogNorm(),vmin=1,vmax=30000)
    points=tax.scatter(times[::step],L[::step],c=density[::step],edgecolors='none',cmap=cmap,norm=matplotlib.colors.LogNorm(),vmin=1,vmax=30000)

def update_tax(t):
    if update_tax.tline is not None:
        update_tax.tline.remove()
    update_tax.tline=tax.axvline(t,color='y')
    
update_tax.tline=None
update_tax.hline=None

def update_polar(t):
    pax.clear()
    ia=np.searchsorted(otimes,t)
    ib=np.searchsorted(otimesb,t)
    i=ia
#    fitresult=emfisis_fit(timesb[i],LL,MLTg,MLATb[i],InvLatb[i]).reshape(LL.shape)
    fitresult=emfisis_fit(times[i],LL,MLTg,MLAT[i],InvLat[i]).reshape(LL.shape)
    im=pax.pcolormesh(MLTg*np.pi/12,LL,fitresult,norm=matplotlib.colors.LogNorm(),cmap=cmap,clim=(1,30000),vmin=1,vmax=30000)
    taillen=1000
    if i>0:
        if ib>0 and ib<len(densityb):
            points=pax.scatter(MLTb[max(ib-taillen,0):ib]*np.pi/12,Lb[max(ib-taillen,0):ib],c=densityb[max(ib-taillen,0):ib],edgecolors='none',cmap=cmap,norm=matplotlib.colors.LogNorm(),vmin=1,vmax=30000)
        points=pax.scatter(MLT[max(ia-taillen,0):ia]*np.pi/12,L[max(ia-taillen,0):ia],c=density[max(ia-taillen,0):ia],edgecolors='none',cmap=cmap,norm=matplotlib.colors.LogNorm(),vmin=1,vmax=30000)
    if ib>0 and ib<len(densityb):
        points=pax.scatter(MLTb[ib]*np.pi/12,Lb[ib],c=densityb[ib],edgecolors='k',cmap=cmap,norm=matplotlib.colors.LogNorm(),vmin=1,vmax=30000)
    if not density.mask[ia]:
        points=pax.scatter(MLT[ia]*np.pi/12,L[ia],c=density[ia],edgecolors='k',cmap=cmap,norm=matplotlib.colors.LogNorm(),vmin=1,vmax=30000)
    if update_polar.cbar is None:
        try:
            update_polar.cbar=fig.colorbar(im)
        except UnboundLocalError:
            pass

def update_fitax():
    fitax.clear()
    inds=np.where((otimes>fitax.tmin) * (otimes<fitax.tmax))
    fitdensity,fituncert,i=emfisis_fit(times[inds],L[inds],MLT[inds],MLAT[inds],InvLat[inds],returnFull=True)
    fitax.plot(L[inds],density[inds],linestyle='',marker='.')
    fitax.plot(L[inds],fitdensity,color='g')
    fitax.plot(L[inds],fitdensity*(fituncert),linestyle=':',color='g')
    fitax.plot(L[inds],fitdensity/(fituncert),linestyle=':',color='g')
    fitax.set_yscale('log')
    fitax.set_ylim(1e0,1e4)
    fitax.set_ylabel(r'Electron density (cm^-3)')
    fitax.set_xlabel('L shell')
    if update_fitax.hline is not None:
        [line.remove() for line in update_fitax.hline]
    update_fitax.hline=tax.plot([fitax.tmin,fitax.tmax],[1.6,1.6],color='y',marker='.')

update_fitax.hline=None
    
fitax.tmin=otimes[0]
fitax.tmax=otimes[-1]

update_polar.cbar=None

def update(t):
    update_polar(t)
    update_tax(t)
    update_fitax()
    fig.canvas.draw()

def animate():
    i=animate.i
    step=30
    if i+step>=len(times):
        return False
    update_polar(i)
    animate.i+=step
    return True

def motion_notify_callback(event):
    'on mouse movement'
    if event.inaxes is not tax: return
    if event.button == 1:
        t,L = event.xdata, event.ydata
        update(t)
    else:
        return

def button_press_callback(event):
    'on button press'
    if event.inaxes is not tax: return
    if event.button == 1:
        t,L = event.xdata, event.ydata
        update(t)
    elif event.button==2:
        t,L = event.xdata, event.ydata
        fitax.tmin=t
        update_fitax()
        fig.canvas.draw()

def button_release_callback(event):
    'on button release'
    if event.inaxes is not tax: return
    if event.button == 1:
        t,L = event.xdata, event.ydata
        update(t)
    elif event.button==2:
        t,L = event.xdata, event.ydata
        fitax.tmax=t
        update_fitax()
        fig.canvas.draw()

fig.canvas.mpl_connect('motion_notify_event',motion_notify_callback)

fig.canvas.mpl_connect('button_press_event',button_press_callback)
fig.canvas.mpl_connect('button_release_event',button_release_callback)

animate.i=0
initialize_tax()
update(otimes[0])

plt.show()
