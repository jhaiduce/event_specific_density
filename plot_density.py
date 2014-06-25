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

times,Lstar,MLT,MLAT,InvLat,density=get_density_and_time('rbspa',datetime(2012,10,6),datetime(2012,10,9))

# Find the points that are valid in all arrays
validpoints=np.where(-(density.mask+times.mask))

# Remove invalid points from all the arrays
times=times[validpoints]
Lstar=Lstar[validpoints]
MLT=MLT[validpoints]
MLAT=MLAT[validpoints]
InvLat=InvLat[validpoints]
density=density[validpoints]

def local_maxima(a):
    return np.r_[True, a[1:] < a[:-1]] & np.r_[a[:-1] < a[1:], True]

maxima=np.where(local_maxima(Lstar))[0]
minima=np.where(local_maxima(-Lstar))[0]

segmentbounds=np.insert(maxima,np.searchsorted(maxima,minima),minima)
segmentbounds[-1]-=1

otimes=date2num(times)

window_len=41
smoothdens=smooth(density,window_len)[window_len/2:-window_len/2+1]
cdiff=(smoothdens[2:]-smoothdens[0:-2])/(Lstar[2:]-Lstar[:-2])

colors=['b','r','g','y','c','m']

fitcoeffs=np.zeros((len(segmentbounds)-1,6))

fig1=plt.figure(figsize=(21,6))
fig2=plt.figure()

for i in range(len(segmentbounds)-1):
    Lseg=Lstar[segmentbounds[i]:segmentbounds[i+1]]
    dseg=density[segmentbounds[i]:segmentbounds[i+1]]
    MLTseg=MLT[segmentbounds[i]:segmentbounds[i+1]]
    MLATseg=MLAT[segmentbounds[i]:segmentbounds[i+1]]
    InvLatseg=InvLat[segmentbounds[i]:segmentbounds[i+1]]
    tseg=times[segmentbounds[i]:segmentbounds[i+1]]
    fig1.gca().plot(tseg,dseg,linestyle='',marker='.',color=colors[i%len(colors)])
    fig2.gca().plot(Lseg,dseg,linestyle='',marker='.',color=colors[i%len(colors)])

    fitresult = leastsq(fitfunc,[3.6,0.8,1,1],args=(Lseg,MLTseg,MLATseg,InvLatseg,dseg),
                        maxfev=10000,full_output=True,ftol=1e-4,xtol=1e-4)
    #print fitresult[2]['nfev']
    pL,pW,ps,ts=fitresult[0]
    fitcoeffs[i,:]=(date2num(tseg[0]),date2num(tseg[-1]),pL,pW,ps,ts)

    fig1.gca().plot(tseg,fitdensity(Lseg,MLTseg,MLATseg,InvLatseg,pL,pW,ps,ts),color='k')
    fig1.gca().plot(tseg,fitdensity(Lseg,MLTseg,MLATseg,InvLatseg,pL,pW,ps,ts)+fituncert(Lseg,MLTseg,MLATseg,InvLatseg,pL,pW,ps,ts,1),color='k',linestyle=':')
    fig1.gca().plot(tseg,fitdensity(Lseg,MLTseg,MLATseg,InvLatseg,pL,pW,ps,ts)-fituncert(Lseg,MLTseg,MLATseg,InvLatseg,pL,pW,ps,ts,-1),color='k',linestyle=':')
    fig2.gca().plot(Lseg,fitdensity(Lseg,MLTseg,MLATseg,InvLatseg,pL,pW,ps,ts),color='k')
    fig2.gca().plot(Lseg,fitdensity(Lseg,MLTseg,MLATseg,InvLatseg,pL,pW,ps,ts)+fituncert(Lseg,MLTseg,MLATseg,InvLatseg,pL,pW,ps,ts,1),color='k',linestyle=':')
    fig2.gca().plot(Lseg,fitdensity(Lseg,MLTseg,MLATseg,InvLatseg,pL,pW,ps,ts)-fituncert(Lseg,MLTseg,MLATseg,InvLatseg,pL,pW,ps,ts,-1),color='k',linestyle=':')

    fig2.gca().set_yscale('log')
    fig2.gca().set_ylim(1e0,1e4)
    fig2.gca().set_ylabel(r'Electron density (cm^-3)')
    fig2.gca().set_xlabel('L (Re)')

fh=open("fitcoeffs.csv",'w')
fh.write("Interval start (ordinal days), Interval end(ordinal days), Plasmapause L* (Re), Plasmapause width (Re), Plasmasphere multiplier, Trough multiplier")
np.savetxt(fh,fitcoeffs,delimiter=',')
fh.close()

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

#plt.plot(times,sheeley_density_trough(Lstar,MLT),label='Normal')
#plt.plot(times,124*(3.0/Lstar)**4.0,label='Without MLT dependence')
plt.plot(times,(36*(3.0/Lstar)**3.5*np.cos((MLT-7.7*(3.0/Lstar)**2.0+12)*np.pi/12))/sheeley_density_trough(Lstar,MLT))
plt.legend()
plt.xlabel('Universal time')
plt.ylabel('Sheeley trough density')

plt.show()
