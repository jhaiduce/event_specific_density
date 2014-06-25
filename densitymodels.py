import emfisis
import ephemeris
from matplotlib import pyplot as plt
from datetime import datetime, timedelta
from matplotlib.dates import date2num, num2date
import numpy as np
from smooth import smooth
from scipy.optimize import leastsq
import matplotlib

#matplotlib.rc('text',usetex=True)

def carpenter_density_plasmasphere(x,R,L):
    return np.exp((-0.3145*L+3.9043)+(0.15*(np.cos(x)-0.5*np.cos(2*x))+0.00127*R-0.0635)*np.exp(-(L-2)/1.5)*np.exp(-(L-2)/1.5))

def sheeley_density_plasmasphere(L):
    return 1390*(3.0/L)**4.83

def sheeley_density_trough(L,LT):
    return 124*(3.0/L)**4.0+36*(3.0/L)**3.5*np.cos((LT-7.7*(3.0/L)**2.0+12)*np.pi/12)

def sheeley_uncertainty_plasmasphere(L):
    return 440*(3.0/L)**3.60

def sheeley_uncertainty_trough(L,LT):
    return 78*(3.0/L)**4.72+17*(3.0/L)**3.75*np.cos((LT-22)*np.pi/12)

def ozhogin_density_equator(L):
    return 10**(4.4693-0.4903)*L

def ozhogin_uncertainty_equator(L,sign):
    return 10**(4.4693+sign*0.0921-(0.4903+sign*0.0315)*L)-ozhogin_density_eq(L)

def ozhogin_density_latitude_factor(lat,lat_inv):
    return np.cos(np.pi/2*1.01*lat/lat_inv)**-0.75

def ozhogin_latitude_factor_uncertainty(lat,lat_inv,sign):
    return np.cos(np.pi/2*(1.01+sign*0.03)*lat/lat_inv)**-(0.75+sign*0.08)-ozhogin_density_latitude_factor(lat,lat_inv)

def smoothstep(edge0,edge1,x):
    x=np.clip((x-edge0)/(edge1-edge0),0,1)
    return x*x*(3-2*x)

def smootherstep(edge0,edge1,x):
    x=np.clip((x-edge0)/(edge1-edge0),0,1)
    return x*x*x*(x*(x*6-15)+10)

def fitdensity(L,MLT,MLAT,InvLat,pL,pW,ps,ts):
    """
    A plasmapause fit function based on the Sheeley model for the trough and plasmasphere

    L: Field line L
    MLT: Magnetic local time
    pL: Plasmapause L
    pW: Plasmapause width (Re)
    ps: Plasmasphere region scaling factor
    ts: Trough region scaling factor
    """

    pDens=sheeley_density_plasmasphere(L)
    tDens=sheeley_density_trough(L,MLT)

    w=smoothstep(pL-pW/2,pL+pW/2,L)

    return (tDens*ts*w + pDens*ps*(1-w))*ozhogin_density_latitude_factor(MLAT,InvLat)

def fituncert(L,MLT,MLAT,InvLat,pL,pW,ps,ts,sign):

    # Sheeley uncertainty
    pUncert=sheeley_uncertainty_plasmasphere(L)
    tUncert=sheeley_uncertainty_trough(L,MLT)

    # Sheeley density
    pDens=sheeley_density_plasmasphere(L)
    tDens=sheeley_density_trough(L,MLT)

    # Plasmapause transition factor
    w=smoothstep(pL-pW/2,pL+pW/2,L)

    # Total uncertainty
    return ((tUncert*ts*w + pUncert*ps*(1-w))/(tDens*ts*w + pDens*ps*(1-w))+ozhogin_latitude_factor_uncertainty(MLAT,InvLat,sign)/ozhogin_density_latitude_factor(MLAT,InvLat))*(tDens*ts*w + pDens*ps*(1-w))*ozhogin_density_latitude_factor(MLAT,InvLat)
    

def fitfunc(x,L,MLT,MLAT,InvLat,meas_dens):
    pL,pW,ps,ts=x
    return np.log(fitdensity(L,MLT,MLAT,InvLat,pL,pW,ps,ts))-np.log(meas_dens)

def get_density_and_time(scname,dstart,dend):
    times,density=emfisis.get_data(scname,['Epoch','density'],dstart,dend)
    #density=emfisis.get_data(scname,'density',dstart,dend)
    ephem=ephemeris.ephemeris(scname,dstart,dend+timedelta(1))
    Lsint=ephem.get_interpolator('Lstar')
    MLTint=ephem.get_interpolator('EDMAG_MLT')
    MLATint=ephem.get_interpolator('EDMAG_MLAT')
    InvLatint=ephem.get_interpolator('InvLat')
    otimes=date2num(times)
    return times,Lsint(otimes),MLTint(otimes),MLATint(otimes),InvLatint(otimes),density

def local_maxima(a):
    return np.r_[True, a[1:] < a[:-1]] & np.r_[a[:-1] < a[1:], True]

class emfisis_fit_model(object):

    def __init__(self,scname):
        self.fitcoeffs=None
        self.scname=scname

    def _calculate_fitcoeffs(self,dates):
        dates=np.array(dates)
        dates=dates[np.argsort(dates)]
        times,Lstar,MLT,MLAT,InvLat,density=get_density_and_time(self.scname,dates[0],dates[-1]+timedelta(1))

        # Find the points that are valid in all arrays
        validpoints=np.where(-(density.mask+times.mask))

        # Remove invalid points from all the arrays
        times=times[validpoints]
        Lstar=Lstar[validpoints]
        MLT=MLT[validpoints]
        MLAT=MLAT[validpoints]
        InvLat=InvLat[validpoints]
        density=density[validpoints]

        maxima=np.where(local_maxima(Lstar))[0]
        minima=np.where(local_maxima(-Lstar))[0]

        segmentbounds=np.insert(maxima,np.searchsorted(maxima,minima),minima)
        segmentbounds[-1]-=1

        otimes=date2num(times)

        window_len=41
        smoothdens=smooth(density,window_len)[window_len/2:-window_len/2+1]
        cdiff=(smoothdens[2:]-smoothdens[0:-2])/(Lstar[2:]-Lstar[:-2])

        fitcoeffs=np.zeros((len(segmentbounds)-1,6))

        for i in range(len(segmentbounds)-1):
            Lseg=Lstar[segmentbounds[i]:segmentbounds[i+1]]
            dseg=density[segmentbounds[i]:segmentbounds[i+1]]
            MLTseg=MLT[segmentbounds[i]:segmentbounds[i+1]]
            MLATseg=MLAT[segmentbounds[i]:segmentbounds[i+1]]
            InvLatseg=InvLat[segmentbounds[i]:segmentbounds[i+1]]
            tseg=times[segmentbounds[i]:segmentbounds[i+1]]

            fitresult = leastsq(fitfunc,[3.6,0.8,1,1],args=(Lseg,MLTseg,MLATseg,InvLatseg,dseg),
                                maxfev=10000,full_output=True,ftol=1e-4,xtol=1e-4)
            #print fitresult[2]['nfev']
            pL,pW,ps,ts=fitresult[0]
            fitcoeffs[i,:]=(date2num(tseg[0]),date2num(tseg[-1]),pL,pW,ps,ts)

        return fitcoeffs

    def __call__(self,datetimes,L,MLT,MLAT,InvLat):
        try:
            _ = (d for d in datetimes)
        except TypeError:
            datetimes=[datetimes]

        fitcoeffs=self.get_fitcoeffs(datetimes)
        if fitcoeffs is None: return None

        print fitcoeffs.shape

        pL,pW,ps,ts=fitcoeffs[:,2:].transpose()
            
        return fitdensity(L,MLT,MLAT,InvLat,pL,pW,ps,ts)

    def search_fitcoeffs(self,odate):
        i=np.searchsorted(self.fitcoeffs[:,0],odate)
        fitcoeffs=np.zeros((len(odate),6))
        fitcoeffs[(i>0),:] = self.fitcoeffs[i[(i>0)]-1,:]
        fitcoeffs[i==1,2:] = None
        fitcoeffs[(self.fitcoeffs[i-1,0]>=odate) + (odate>=self.fitcoeffs[i-1,1]),2:]=None
        return fitcoeffs

    def get_fitcoeffs(self,dates):
        odate=date2num(dates)

        if self.fitcoeffs is None:
            self.fitcoeffs=self._calculate_fitcoeffs(dates)

        print odate, self.fitcoeffs.shape

        fitcoeffs=self.search_fitcoeffs(odate)
        if not np.isnan(fitcoeffs).any(): return fitcoeffs

        fitcoeffs=self._calculate_fitcoeffs(dates[np.where(np.isnan(fitcoeffs[:,2]))])
        try:
            inds=np.searchsorted(self.fitcoeffs[:,0],fitcoeffs[:,0])
            self.fitcoeffs=np.insert(self.fitcoeffs,inds,fitcoeffs,axis=0)
        except TypeError:
            self.fitcoeffs=fitcoeffs

        return self.search_fitcoeffs(odate)
