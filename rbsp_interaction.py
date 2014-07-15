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
from matplotlib.patches import Rectangle
import themisdata

times,L,MLT,MLAT,InvLat,density=get_density_and_time('rbspa',datetime(2012,10,6),datetime(2012,10,10))
otimes=date2num(times)

timesb,Lb,MLTb,MLATb,InvLatb,densityb=get_density_and_time('rbspb',datetime(2012,10,8),datetime(2012,10,9))
otimesb=date2num(timesb)

emfisis_fit=emfisis_fit_model('rbspa')

show_themis=False

if show_themis:
    themis_data=themisdata.load()
    themis_data.sort(order=('time'))
    themis_data=themis_data[:][(themis_data['time']>times[0])*(themis_data['time']<times[-1])]
    themis_data=themis_data[:][(themis_data['L']>1.5)*(themis_data['L']<6.0)]
    themis_otimes=date2num(themis_data['time'])

fig=plt.figure()
pax=plt.subplot(2,2,1,polar=True)
tax=plt.subplot(2,2,3)
mltax=plt.subplot(2,2,4)
fitax=plt.subplot(2,2,2)
cmap=plt.get_cmap('spectral')
Llin=np.linspace(1.5,6,100)
MLTlin=np.linspace(0,24,100)
MLTg,LL=np.meshgrid(MLTlin,Llin)
#Tlin=np.linspace(date2num(times[0]),date2num(times[0])+2,100)
Tlin=np.linspace(date2num(times[0]),date2num(times[-1]),100)
T,LL=np.meshgrid(Tlin,Llin)

def initialize_tax():
    tax.clear()
    step=10
    fitresult=emfisis_fit(T,LL,MLT=0,MLAT=0,InvLat=1).reshape(LL.shape)
    im=tax.imshow(fitresult,origin='lower',extent=(Tlin.min(),Tlin.max(),Llin.min(),Llin.max()),aspect='auto',norm=matplotlib.colors.LogNorm(),clim=(1,30000))
    points=tax.scatter(timesb[::step],Lb[::step],c=densityb[::step],edgecolors='none',cmap=cmap,norm=matplotlib.colors.LogNorm(),vmin=1,vmax=30000)
    points=tax.scatter(times[::step],L[::step],c=density[::step],edgecolors='none',cmap=cmap,norm=matplotlib.colors.LogNorm(),vmin=1,vmax=30000)
    if show_themis:
        for spacecraft in (1,4,5):
            inds=np.where(themis_data['scnumber']==spacecraft)
            MLTt=themis_data['MLT'][inds]
            Lt=themis_data['L'][inds]
            timest=themis_data['time'][inds]
            densityt=themis_data['density'][inds]
            points=tax.scatter(timest[::step],Lt[::step],c=densityt[::step],edgecolors='none',cmap=cmap,norm=matplotlib.colors.LogNorm(),vmin=1,vmax=30000)

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
    step=10
#    fitresult=emfisis_fit(timesb[i],LL,MLTg,MLATb[i],InvLatb[i]).reshape(LL.shape)
    fitresult=emfisis_fit(times[i],LL,MLTg,MLAT[i],InvLat[i]).reshape(LL.shape)
    im=pax.pcolormesh(-MLTg*np.pi/12-np.pi/2,LL,fitresult,norm=matplotlib.colors.LogNorm(),cmap=cmap,clim=(1,30000),vmin=1,vmax=30000)
    taillen=1000
    if i>0:

        if ib>0 and ib<len(densityb):
            points=pax.scatter(
                -MLTb[max(ib-taillen,0):ib:step]*np.pi/12-np.pi/2,
                 Lb[max(ib-taillen,0):ib:step],
                 c=densityb[max(ib-taillen,0):ib:step],
                 edgecolors='none',cmap=cmap,norm=matplotlib.colors.LogNorm(),vmin=1,vmax=30000)

        points=pax.scatter(
            -MLT[max(ia-taillen,0):ia:step]*np.pi/12-np.pi/2,
             L[max(ia-taillen,0):ia:step],
             c=density[max(ia-taillen,0):ia:step],
             edgecolors='none',cmap=cmap,norm=matplotlib.colors.LogNorm(),vmin=1,vmax=30000)

    if ib>0 and ib<len(densityb):
        points=pax.scatter(
            -MLTb[ib]*np.pi/12-np.pi/2,
             Lb[ib],
             c=densityb[ib],
             edgecolors='k',cmap=cmap,norm=matplotlib.colors.LogNorm(),vmin=1,vmax=30000)

    if not density.mask[ia]:
        points=pax.scatter(
            -MLT[ia]*np.pi/12-np.pi/2,
             L[ia],
             c=density[ia],
             edgecolors='k',cmap=cmap,norm=matplotlib.colors.LogNorm(),vmin=1,vmax=30000)
    if show_themis:
        for spacecraft in (1,4,5):
            inds=np.where(themis_data['scnumber']==spacecraft)
            it=np.searchsorted(themis_otimes[inds],t)
            if it==0 or it>=len(inds[0]): 
                continue
            MLTt=themis_data['MLT'][inds]
            Lt=themis_data['L'][inds]
            densityt=themis_data['density'][inds]
            points=pax.scatter(-MLTt[max(it-taillen,0):it:step]*np.pi/12-np.pi/2,Lt[max(it-taillen,0):it:step],c=densityt[max(it-taillen,0):it:step],edgecolors='none',cmap=cmap,norm=matplotlib.colors.LogNorm(),vmin=1,vmax=30000)
            points=pax.scatter(-MLTt[it]*np.pi/12-np.pi/2,Lt[it],c=densityt[it],edgecolors='k',cmap=cmap,norm=matplotlib.colors.LogNorm(),vmin=1,vmax=30000)
        pax.set_xticklabels(['18','15','12','09','06','03','24','21'])
        
    if update_polar.cbar is None:
        try:
            update_polar.cbar=fig.colorbar(im)
        except UnboundLocalError:
            pass

def update_fitax():
    fitax.clear()

    inds=np.where((otimes>fitax.tmin) * (otimes<fitax.tmax) * (L>fitax.lmin) * (L<fitax.lmax))
    fitdensity,fituncert,i=emfisis_fit(times[inds],L[inds],MLT[inds],MLAT[inds],InvLat[inds],returnFull=True)
    fitax.plot(L[inds],density[inds],linestyle='',marker='.')
    fitax.plot(L[inds],fitdensity,color='g')
    fitax.plot(L[inds],fitdensity*(fituncert),linestyle=':',color='g')
    fitax.plot(L[inds],fitdensity/(fituncert),linestyle=':',color='g')

    # RBSP-b
    inds=np.where((otimesb>fitax.tmin) * (otimesb<fitax.tmax) * (Lb>fitax.lmin) * (Lb<fitax.lmax))
    fitdensity,fituncert,i=emfisis_fit(timesb[inds],Lb[inds],MLTb[inds],MLATb[inds],InvLatb[inds],returnFull=True)
    fitax.plot(Lb[inds],densityb[inds],linestyle='',marker='.',color='g')
    fitax.plot(Lb[inds],fitdensity,linestyle='-',color='g')
    fitax.plot(Lb[inds],fitdensity*(fituncert),linestyle=':',color='g')
    fitax.plot(Lb[inds],fitdensity/(fituncert),linestyle=':',color='g')

    # Themis
    if show_themis:
        for spacecraft in (1,4,5):
            inds=np.where((themis_data['scnumber']==spacecraft) 
                          * (themis_data['L']>fitax.lmin) * (themis_data['L']<fitax.lmax)
                          * (themis_otimes>fitax.tmin) * (themis_otimes<fitax.tmax))
            fitdensity,fituncert,i=emfisis_fit(themis_data['time'][inds],themis_data['L'][inds],themis_data['MLT'][inds],MLAT=0,InvLat=1,returnFull=True)
            fitax.plot(themis_data['L'][inds],themis_data['density'][inds],linestyle='',marker='.')
            fitax.plot(themis_data['L'][inds],fitdensity,linestyle='-',color='g')
            fitax.plot(themis_data['L'][inds],fitdensity*(fituncert),linestyle=':',color='g')
            fitax.plot(themis_data['L'][inds],fitdensity/(fituncert),linestyle=':',color='g')

    fitax.set_yscale('log')
    fitax.set_ylim(1e0,1e4)
    fitax.set_ylabel(r'Electron density (cm^-3)')
    fitax.set_xlabel('L shell')

def update_fitax_limits():
    if update_fitax.hline is not None:
        update_fitax.hline.remove()
    #    [line.remove() for line in update_fitax.hline]
    #update_fitax.hline=tax.plot([fitax.tmin,fitax.tmax],[1.6,1.6],color='y',marker='.')
    update_fitax.hline = tax.add_patch(Rectangle((fitax.tmin,fitax.lmin),(fitax.tmax-fitax.tmin),(fitax.lmax-fitax.lmin),facecolor='y',alpha=0.5))

def update_mltax():
    mltax.clear()
    
    # RBSP-a
    inds=np.where((otimes>fitax.tmin) * (otimes<fitax.tmax) * (L>fitax.lmin) * (L<fitax.lmax))
    fitdensity,fituncert,i=emfisis_fit(times[inds],L[inds],MLT[inds],MLAT[inds],InvLat[inds],returnFull=True)
    mltax.plot(MLT[inds],np.abs(density[inds]-fitdensity),linestyle='',marker='.')
    #mltax.plot(MLT[inds],fitdensity,color='g')
    #mltax.plot(MLT[inds],fitdensity*(fituncert),linestyle=':',color='g')
    #mltax.plot(MLT[inds],fitdensity/(fituncert),linestyle=':',color='g')

    # RBSP-b
    inds=np.where((otimesb>fitax.tmin) * (otimesb<fitax.tmax) * (Lb>fitax.lmin) * (Lb<fitax.lmax))
    fitdensity,fituncert,i=emfisis_fit(timesb[inds],Lb[inds],MLTb[inds],MLATb[inds],InvLatb[inds],returnFull=True)
    mltax.plot(MLTb[inds],np.abs(densityb[inds]-fitdensity),linestyle='',marker='.',color='g')

    # Themis
    if show_themis:
        for spacecraft in (1,4,5):
            inds=np.where((themis_data['scnumber']==spacecraft) 
                          * (themis_data['L']>fitax.lmin) * (themis_data['L']<fitax.lmax)
                          * (themis_otimes>fitax.tmin) * (themis_otimes<fitax.tmax))
            fitdensity,fituncert,i=emfisis_fit(themis_data['time'][inds],themis_data['L'][inds],themis_data['MLT'][inds],MLAT=0,InvLat=1,returnFull=True)
            mltax.plot(themis_data['MLT'][inds],np.abs(themis_data['density'][inds]-fitdensity),linestyle='',marker='.')

    mltax.set_yscale('log')
    #mltax.set_ylim(1e0,1e4)
    mltax.set_ylabel(r'Electron density (cm^-3)')
    mltax.set_xlabel('MLT')

update_fitax.hline=None
    
fitax.tmin=otimes[0]
fitax.tmax=otimes[-1]
fitax.lmin=1.5
fitax.lmax=6

update_polar.cbar=None

def update(t):
    update_polar(t)
    update_tax(t)
    update_fitax()
    update_mltax()
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
    elif event.button==2:
        t,L = event.xdata, event.ydata
        fitax.tmax=t
        fitax.lmax=L
        update_fitax_limits()
        fig.canvas.draw()
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
        fitax.lmin=L
        update_fitax_limits()
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
        fitax.lmax=L
        update_fitax()
        update_mltax()
        fig.canvas.draw()

fig.canvas.mpl_connect('motion_notify_event',motion_notify_callback)

fig.canvas.mpl_connect('button_press_event',button_press_callback)
fig.canvas.mpl_connect('button_release_event',button_release_callback)

animate.i=0
initialize_tax()
update(otimes[0])

plt.show()
