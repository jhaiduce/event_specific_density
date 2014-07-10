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

class mlt_comparator(object):

    def __init__(self,fig):

        times,L,MLT,MLAT,InvLat,density=get_density_and_time('rbspa',datetime(2012,10,6),datetime(2012,10,10))
        otimes=date2num(times)

        timesb,Lb,MLTb,MLATb,InvLatb,densityb=get_density_and_time('rbspb',datetime(2012,10,8),datetime(2012,10,9))
        otimesb=date2num(timesb)

        self.L=L
        self.times=times
        self.otimes=otimes
        self.density=density

        self.emfisis_fit=emfisis_fit_model('rbspa')
        fitdensity,fituncert,inds=self.emfisis_fit(times,L,MLT,MLAT,InvLat,returnFull=True)
        self.fitdensity=fitdensity

        self.fig=fig
        self.ax1=plt.subplot(2,1,1)
        self.ax2=plt.subplot(2,1,2)

        self.ax2.plot(L,density,linestyle='',marker='.')

        self.Lline=None
        self.Llim=np.array([1.5,2])
        self.pressed=False

        self.uinds=np.unique(inds)
        self.uinds=self.uinds[self.uinds<self.emfisis_fit.fitcoeffs.shape[0]]

        self.fig.canvas.mpl_connect('motion_notify_event',self.select_L)
        self.fig.canvas.mpl_connect('button_press_event',self.select_Lstart)
        self.fig.canvas.mpl_connect('button_release_event',self.on_release)

    def update_L(self,Llim):
        self.ax1.clear()
        lmin,lmax=np.sort(Llim)

        otimes=self.otimes
        L=self.L
        x1avg=np.zeros((len(self.uinds)/2))
        x2avg=np.zeros((len(self.uinds)/2))
        x1std=np.zeros((len(self.uinds)/2))
        x2std=np.zeros((len(self.uinds)/2))
        for ind in self.uinds:
            if ind%2==0: continue
            inL=(L>lmin)*(L<lmax)
            thispass=(otimes>self.emfisis_fit.fitcoeffs[ind,0])*(otimes<>self.emfisis_fit.fitcoeffs[ind,1])
            nextpass=(otimes>self.emfisis_fit.fitcoeffs[ind+1,0])*(otimes<>self.emfisis_fit.fitcoeffs[ind+1,1])
            ind1=np.where(thispass*inL)
            ind2=np.where(nextpass*inL)
            x1=self.density[ind1]/self.fitdensity[ind1]
            x2=self.density[ind2]/self.fitdensity[ind2]

            if len(ind1)==0:
                x1avg[ind/2]=None
            else: 
                x1avg[ind/2]=np.ma.average(x1)
            if len(ind2)==0: 
                x2avg[ind/2]=None
            else:
                x2avg[ind/2]=np.ma.average(x2)
            x1std[ind/2]=np.ma.std(x1)
            x2std[ind/2]=np.ma.std(x2)
        self.ax1.plot(x1avg,x2avg,linestyle='',marker='.',color='b')
        #self.ax1.errorbar(x1avg,x2avg,x1std,x2std,linestyle='',marker='.',color='b')

    def show_L(self,Llim):
        lmin,lmax=np.sort(Llim)
        if self.Lline is not None:
            self.Lline.remove()
        self.Lline=self.ax2.axvspan(lmin,lmax,facecolor='y',alpha=0.5)

    def select_L(self,event):
        'on mouse movement'
        if event.inaxes is not self.ax2: return
        if event.button == 1 and self.pressed:
            L,n = event.xdata, event.ydata
            self.Llim[1]=L
            self.show_L(self.Llim)
            self.fig.canvas.draw()
        else:
            return

    def on_release(self,event):
        if event.inaxes is not self.ax2: return
        if event.button == 1:
            self.pressed=False
            L,n = event.xdata, event.ydata
            self.Llim[1]=L
            self.show_L(self.Llim)
            self.update_L(self.Llim)
            self.fig.canvas.draw()
        else:
            return

    def select_Lstart(self,event):
        'on mouse press'
        if event.inaxes is not self.ax2: return
        if event.button == 1:
            self.pressed=True
            L,n = event.xdata, event.ydata
            self.Llim[0]=L
        else:
            return

fig=plt.figure()

mlt_comparator(fig)

plt.show()

