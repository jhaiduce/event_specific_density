import emfisis
import ephemeris
from matplotlib import pyplot as plt
from datetime import datetime, timedelta
from matplotlib.dates import date2num, num2date
import numpy as np
from smooth import smooth
from scipy.optimize import leastsq
import matplotlib
from scipy.interpolate import Rbf, UnivariateSpline, interp1d

#matplotlib.rc('text',usetex=True)

def carpenter_density_plasmasphere(x,R,L):
    """
    Carpenter plasmasphere density model
    
    Args:
        x (float): Given by $x = 2\pi(d+9)/365$, where d is the day number.

        R (float): 13-month average sunspot number

        L (float): L-shell

    Returns:
        density (float)
    """
    return np.exp((-0.3145*L+3.9043)+(0.15*(np.cos(x)-0.5*np.cos(2*x))+0.00127*R-0.0635)*np.exp(-(L-2)/1.5)*np.exp(-(L-2)/1.5))

def sheeley_density_plasmasphere(L):
    """
    Sheeley plasmasphere density model

    Args:
        L (float): L-shell

    Returns:
        density (float)
    """
    return 1390*(3.0/L)**4.83

def sheeley_density_trough(L,LT):
    """
    Sheeley trough density model

    Args:
        L (float): L-shell

        LT (float): Local time in hours (0-24)

    Returns:
        density (float)
    """
    return 124*(3.0/L)**4.0+36*(3.0/L)**3.5*np.cos((LT-7.7*(3.0/L)**2.0+12)*np.pi/12)

def sheeley_uncertainty_plasmasphere(L):
    """
    Uncertainty corresponding to the Sheeley plasmasphere density model

    Args:
        L (float): L-shell

    Returns:
        Density uncertainty (float)
    """
    return 440*(3.0/L)**3.60

def sheeley_uncertainty_trough(L,LT):
    """
    Uncertainty corresponding to the Sheeley trough density model

    Args:
        L (float): L-shell

        LT (float): Local time in hours (0-24)

    Returns:
        Density uncertainty (float)
    """
    return 78*(3.0/L)**4.72+17*(3.0/L)**3.75*np.cos((LT-22)*np.pi/12)

def ozhogin_density_equator(L):
    """
    Ozhogin equatorial density model

    Args:
        L (float): L-shell

    Returns:
        density (float)
    """
    return 10**(4.4693-0.4903)*L

def ozhogin_uncertainty_equator(L,sign):
    """
    Uncertainty corresponding to the Ozhogin equatorial density model

    Args:
        L (float): L-shell

        sign (number): +1 or -1 indicating whether "upper" or "lower" side of error bar should be computed.

    Returns:
        density uncertainty (float)
    """
    return 10**(4.4693+sign*0.0921-(0.4903+sign*0.0315)*L)-ozhogin_density_eq(L)

def ozhogin_density_latitude_factor(lat,lat_inv):
    """
    This calculates the latitude factor from the Ozhogin model. Returns 1 at the equator, <1 elsewhere.

    Args:
        lat (float): Latitude

        lat_inv (float): Invariant latitude

    Returns:
        float
    """
    return np.cos(np.pi/2*1.01*lat/lat_inv)**-0.75

def ozhogin_latitude_factor_uncertainty(lat,lat_inv,sign):
    """
    This calculates uncertainty of the Ozhogin latitude factor

    Args:
        lat (float): Latitude

        lat_inv (float): Invariant latitude

        sign (number): +1 or -1 indicating whether "upper" or "lower" side of error bar should be computed.

    Returns:
        float
    """
    return np.cos(np.pi/2*(1.01+sign*0.03)*lat/lat_inv)**-(0.75+sign*0.08)-ozhogin_density_latitude_factor(lat,lat_inv)

def smoothstep(edge0,edge1,x):
    """
    A cubic interpolation function going from (edge0,0) to (edge1,1). See http://en.wikipedia.org/wiki/Smoothstep.

    Args:
        edge0 (float): Lower limit of interpolation function (x<edge0 will return 0)

        edge1 (float): Upper limit of interpolation function (x>edge1 will return 1)

        x (float): Value to be interpolated

    Returns:
        float
    """
    x=np.clip((x-edge0)/(edge1-edge0),0,1)
    return x*x*(3-2*x)

def smootherstep(edge0,edge1,x):
    """
    A polynomial interpolation function going from (edge0,0) to (edge1,1). Like smoothstep but has zero 1st and 2nd derivatives at the endpoints. See http://en.wikipedia.org/wiki/Smoothstep.

    Args:
        edge0 (float): Lower limit of interpolation function (x<edge0 will return 0)

        edge1 (float): Upper limit of interpolation function (x>edge1 will return 1)

        x (float): Value to be interpolated

    Returns:
        float
    """
    x=np.clip((x-edge0)/(edge1-edge0),0,1)
    return x*x*x*(x*(x*6-15)+10)

def fitdensity(L,MLT,MLAT,InvLat,pL,pW,ps,ts):
    """
    A plasmapause fit function based on the Sheeley model for the trough and plasmasphere, combined with the Ozhogin model for latitude dependence.

    Args:
        L (float): Field line L.

        MLT (float): Magnetic local time.

        pL (float): Plasmapause L.

        pW (float): Plasmapause width (Re).

        ps (float): Plasmasphere region scaling factor.

        ts (float): Trough region scaling factor.

    Returns:
        Density (float)
    """

    pDens=sheeley_density_plasmasphere(L)
    tDens=sheeley_density_trough(L,MLT)

    pp=pW*(1+0.2571*np.sin(2*np.pi*(MLT-6)/24))

    w=smoothstep(pL-pp/2,pL+pp/2,L)

    return (tDens*ts*w + pDens*ps*(1-w))*ozhogin_density_latitude_factor(MLAT,InvLat)

def fituncert(L,MLT,MLAT,InvLat,pL,pW,ps,ts,sign):
    """
    Uncertainty model corresponding to the function fitdensity.

    Args:
        L (float): Field line L.

        MLT (float): Magnetic local time.

        MLAT (float): Magnetic latitude.

        pL (float): Plasmapause L.

        pW (float): Plasmapause width (Re).

        ps (float): Plasmasphere region scaling factor.

        ts (float): Trough region scaling factor.

        sign (number): +1 or -1 indicating whether "upper" or "lower" side of error bar should be computed.

    Returns:
        Uncertainty (float)
    """

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
    """
    A fit function used to find a best fit of fitdensity() to the passed meas_dens.

    Args:
        x (array-like): An array of (pL, pW, ps, ts) to be passed to fitdensity.

        L (float): Field line L.

        MLT (float): Magnetic local time.

        MLAT (float): Magnetic latitude.

        meas_dens (array-like): Array of density measurements

    Returns:
        
    """
    pL,pW,ps,ts=x
    return np.log(fitdensity(L,MLT,MLAT,InvLat,pL,pW,ps,ts))-np.log(meas_dens)

def get_density_and_time(scname,dstart,dend):
    """
    Get the density measurements from the given spacecraft within the given date range.

    Args:
        scname (str): Spacecraft name (currently 'rbspa' or 'rbspb' are accepted)

        dstart (datetime): Start date

        dend (datetime): End date

    Returns:
        A tuple containing the following arrays:
            - times (array of datetime objects)
            - otimes (array of ordinal times produced by running matplotlib's date2num on the times array)
            - MLT (array of magnetic local time values corresponding to otimes)
            - MLAT (array of magnetic latitudes corresponding to otimes)
            - InvLat (array of invariant latitude corresponding to otimes)
            - density (array of densities corresponding to otimes)
    """
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

class emfisis_density_model(object):

    def _getdata(self,scname,dates):

        dates=np.array(dates)
        dates=dates[np.argsort(dates)]
        try:
            dtend=dates[-1]+timedelta(1)
            dtstart=dates[0]
        except TypeError:
            dtend=num2date(dates[-1]+1)
            dtstart=num2date(dates[0])
        times,Lstar,MLT,MLAT,InvLat,density=get_density_and_time(scname,dtstart,dtend)

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
        return times,Lstar,MLT,MLAT,InvLat,density,segmentbounds
        

class emfisis_smoothing_model(emfisis_density_model):

    """
    Time-dependent electron density model based on RBSP density measurements. Smooths the densities measured by the EMFISIS instrument and extrapolates them into all local times and latitudes.

    Example::

        times, L, MLT, MLAT, InvLat, density = get_density_and_time('rbspa', datetime(2012,10,6), datetime(2012,10,10))
        emfisis_fit = emfisis_fit_model('rbspa')
        fitdensity, fituncert, inds = emfisis_fit(times, L, MLT, MLAT, InvLat, returnFull=True)
    """

    def __init__(self,scname,**kwargs):
        """
        Set-up the density model

        Args:
            scname (str): RBSP spacecraft to use ('rbspa' or 'rbspb')
        """
        self.scname=scname
        self.binwidths=0.5
        self.uncertbins=np.arange(1.5,6.5,self.binwidths)
        self.kwargs=kwargs
        self.segmentbounds=None

    def _create_interpolators(self,dates):
        """
        Load the data from the RBSP spacecraft
        """

        times,Lstar,MLT,MLAT,InvLat,density,segmentbounds=self._getdata(self.scname,dates)

        interpolators=np.zeros((len(segmentbounds)-1,),dtype=object)
        segmentlimits=np.zeros((len(segmentbounds)-1,2))

        for i in range(len(segmentbounds)-1):
            Lseg=Lstar[segmentbounds[i]:segmentbounds[i+1]]
            dseg=density[segmentbounds[i]:segmentbounds[i+1]]
            MLTseg=MLT[segmentbounds[i]:segmentbounds[i+1]]
            MLATseg=MLAT[segmentbounds[i]:segmentbounds[i+1]]
            InvLatseg=InvLat[segmentbounds[i]:segmentbounds[i+1]]
            tseg=times[segmentbounds[i]:segmentbounds[i+1]]

            inds=np.argsort(Lseg)

#            if self.epsilon==0:
#                interpolators[i]=interp1d(Lseg[inds],dseg[inds],bounds_error=False)
#            else:
                #interpolators[i]=Rbf(Lseg,dseg,**self.kwargs)
            interpolators[i]=UnivariateSpline(Lseg[inds],dseg[inds],**self.kwargs)
                
            segmentlimits[i,:]=date2num(tseg[0]),date2num(tseg[-1])

        return segmentlimits,interpolators

    def __call__(self,datetimes,L,MLT,MLAT,InvLat,minDensity=1e-1,returnFull=False):
        """
        Get the density (and optionally uncertainty) at the requested point(s).

        Args:
            datetimes (datetime or array of datetimes): Times to calculate density at

            L (float array): Array of L-shell values

            MLT (float array): Array of magnetic local times
            
            MLAT (float array): Array of magnetic latitudes
            
            InvLat (float array): Array of invariant latitudes

            minDensity (float): Minimum density to return (to prevent returning zero or negative densities

            returnFull (bool): Whether to return the uncertainties along with the densities

        Returns:
        
            Densities at the requested points, or a tuple of (densities, uncertainties).
        """
        try:
            _ = (d for d in datetimes)
        except TypeError:
            datetimes=np.ma.array([datetimes])

        if len(datetimes)==0:
            if returnFull:
                return np.array([]),np.array([]),np.array([])
            else:
                return np.array([])

        try:
            odates=date2num(datetimes)
        except AttributeError:
            odates=datetimes

        # Flatten arrays
        L=np.ma.array(L).flatten()
        MLT=np.ma.array(MLT).flatten()
        odates=odates.flatten()

        if self.segmentbounds is None:
            self.segmentbounds,self.interpolators=self._create_interpolators(odates)

        if odates.max()>self.segmentbounds.max():
            segmentbounds,interpolators=self._create_interpolators([self.segmentbounds.max(),odates.max()])
            self.segmentbounds=np.concatenate((self.segmentbounds,segmentbounds))
            self.interpolators=np.concatenate((self.interpolators,interpolators))
        if odates.min()<self.segmentbounds.min():
            segmentbounds,interpolators=self._create_interpolators([odates.min(),self.segmentbounds.min()])
            self.segmentbounds=np.concatenate((segmentbounds,self.segmentbounds))
            self.interpolators=np.concatenate((interpolators,self.interpolators))

        # Find dates in array
        i=np.searchsorted(self.segmentbounds[:,0],odates)

        # Build array of values matching dates
        densities=np.ma.ones((len(L)))
        densities.fill(np.ma.masked)
        segmentbounds=self.segmentbounds
        in_limits=(i>0)*(i<self.segmentbounds.shape[0])
        if len(odates)==len(L):
            if len(i[in_limits])>0:
                densities[in_limits] = [self.interpolators[k](L[j]) for j,k in enumerate(i[in_limits]-1) ]
        elif len(odates)==1:
            if len(i[in_limits])>0:
                densities = self.interpolators[i[in_limits]-1][0](L)
        else:
            raise ValueError('Size mismatch between datetimes array and L.')

        densities=np.ma.masked_invalid(densities)

        densities[densities<minDensity]=minDensity

        if returnFull:
            return densities,np.zeros(densities.shape),i
        else:
            return densities

class emfisis_fit_model(emfisis_density_model):

    """
    Time-dependent electron density model based on RBSP density measurements. Fits a modified Sheeley density model (see the fitdensity function) to the nearest (in time) density data from the given RBSP spacecraft.

    Example::

        times, L, MLT, MLAT, InvLat, density = get_density_and_time('rbspa', datetime(2012,10,6), datetime(2012,10,10))
        emfisis_fit = emfisis_fit_model('rbspa')
        fitdensity, fituncert, inds = emfisis_fit(times, L, MLT, MLAT, InvLat, returnFull=True)
    """

    def __init__(self,scname):
        """
        Set-up the density model

        Args:
            scname (str): RBSP spacecraft to use ('rbspa' or 'rbspb')
        """
        self.fitcoeffs=None
        self.scname=scname
        self.binwidths=0.5
        self.uncertbins=np.arange(1.5,6.5,self.binwidths)

    def _calculate_fitcoeffs(self,dates):

        times,Lstar,MLT,MLAT,InvLat,density,segmentbounds=self._getdata(self.scname,dates)

        self.segmentbounds=segmentbounds

        fitcoeffs=np.zeros((len(segmentbounds)-1,6))

        fituncert=np.zeros((len(segmentbounds)-1,len(self.uncertbins)+2))
        fituncert.fill(None)

        for i in range(len(segmentbounds)-1):
            Lseg=Lstar[segmentbounds[i]:segmentbounds[i+1]]
            dseg=density[segmentbounds[i]:segmentbounds[i+1]]
            MLTseg=MLT[segmentbounds[i]:segmentbounds[i+1]]
            MLATseg=MLAT[segmentbounds[i]:segmentbounds[i+1]]
            InvLatseg=InvLat[segmentbounds[i]:segmentbounds[i+1]]
            tseg=times[segmentbounds[i]:segmentbounds[i+1]]

            fitresult = leastsq(fitfunc,[3.6,0.8,1,1],args=(Lseg,MLTseg,MLATseg,InvLatseg,dseg),
                                maxfev=10000,full_output=True,ftol=1e-4,xtol=1e-4)
            pL,pW,ps,ts=fitresult[0]
            fitcoeffs[i,:]=(date2num(tseg[0]),date2num(tseg[-1]),pL,pW,ps,ts)
            fituncert[i,0:2]=date2num(tseg[0]),date2num(tseg[-1])

            fitvalues=fitdensity(Lseg,MLTseg,MLATseg,InvLatseg,pL,pW,ps,ts)
            for j in range(len(self.uncertbins)-1):
                binInds=np.where((self.uncertbins[j]<Lseg) * (Lseg<self.uncertbins[j+1]))
                fituncert[i,j+2]=np.exp(np.sqrt(((np.log(fitvalues[binInds])-np.log(dseg[binInds]))**2).sum()/len(binInds[0])))
        return fitcoeffs,fituncert

    def __call__(self,datetimes,L,MLT,MLAT,InvLat,minDensity=1e-1,returnFull=False):
        """
        Get the density (and optionally uncertainty) at the requested point(s).

        Args:
            datetimes (datetime or array of datetimes): Times to calculate density at

            L (float array): Array of L-shell values

            MLT (float array): Array of magnetic local times
            
            MLAT (float array): Array of magnetic latitudes
            
            InvLat (float array): Array of invariant latitudes

            minDensity (float): Minimum density to return (to prevent returning zero or negative densities

            returnFull (bool): Whether to return the uncertainties and segment indices along with the densities

        Returns:
        
            Densities at the requested points, or a tuple of (densities, uncertainties).
        """
        try:
            _ = (d for d in datetimes)
        except TypeError:
            datetimes=np.array([datetimes])

        if len(datetimes)==0:
            if returnFull:
                return [],[],[]
            else:
                return []

        fitcoeffs,fituncert,inds=self.get_fitcoeffs(datetimes,returnFull=True)

        pL,pW,ps,ts=fitcoeffs[:,2:].transpose()

        # Flatten arrays (if possible)
        try:
            L=L.flatten()
            MLT=MLT.flatten()
        except:
            pass

        fitvalues=np.maximum(fitdensity(L,MLT,MLAT,InvLat,pL,pW,ps,ts),minDensity)

        if returnFull:
            uncertinds=np.searchsorted(self.uncertbins,L.flatten())
            uncert=[fituncert[i,uncertinds[i]+1] for i in range(len(uncertinds))]
            return fitvalues,uncert,inds
        else:
            return fitvalues

    def search_fitcoeffs(self,odate,returnInds=False):
        return self.searcharray(odate,self.fitcoeffs,returnInds)

    def searcharray(self,odate,arr,return_inds=False):

        # Find dates in array
        i=np.searchsorted(arr[:,0],odate)

        # Build array of values matching dates
        searchvalues=np.ones((len(odate),arr.shape[1]))
        searchvalues.fill(None)
        searchvalues[(i>0),:] = arr[i[(i>0)]-1,:]
        searchvalues[i==1,2:] = arr[0,2:]

        if return_inds:
            return searchvalues,i
        else:
            return searchvalues

    def get_fitcoeffs(self,dates,returnFull=False):
        dates=dates.flatten()

        try:
            odate=date2num(dates)
        except AttributeError:
            odate=dates

        if self.fitcoeffs is None:
            self.fitcoeffs,self.fituncert=self._calculate_fitcoeffs(dates)

        fitcoeffs,inds=self.search_fitcoeffs(odate,returnInds=True)
        if not np.isnan(fitcoeffs).any(): 
            if returnFull:
                uncert=self.searcharray(odate,self.fituncert)
                return fitcoeffs,uncert,inds
            else:
                return fitcoeffs

        fitcoeffs,fituncert=self._calculate_fitcoeffs(dates[np.where(np.isnan(fitcoeffs[:,2]))])
        try:
            inds=np.searchsorted(self.fitcoeffs[:,0],fitcoeffs[:,0])
            self.fitcoeffs=np.insert(self.fitcoeffs,inds,fitcoeffs,axis=0)
            self.fituncert=np.insert(self.fituncert,inds,fituncert,axis=0)
            values,inds=np.unique(self.fitcoeffs[:,0],return_index=True)
            self.fitcoeffs=self.fitcoeffs[inds,:]
            self.fituncert=self.fituncert[inds,:]
        except TypeError:
            self.fitcoeffs=fitcoeffs

        if returnFull:
            fitcoeffs,inds=self.search_fitcoeffs(odate,returnInds=True)
            uncert=self.searcharray(odate,self.fituncert)
            return fitcoeffs,uncert,inds
        else:
            return self.search_fitcoeffs(odate)
