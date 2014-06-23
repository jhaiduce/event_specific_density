import numpy as np
import h5py
import os
import datetime
from matplotlib.dates import date2num, num2date
from scipy.interpolate import interp1d

class DiscontinuousInterpolator(object):

    """Interpolates a function which increases monotonically but resets to zero upon reaching a defined value"""

    def __init__(self,x,y,period,offset=0):
        y=y-offset
        di=np.where(y[1:]-y[:-1]<0)
        for i in di[0]:
            y[i:]=y[i:]+offset

        self.int=interp1d(x,y)
        self.period=24
        self.offset=offset
    
    def __call__(self,x):
        try:
            _ = (d for d in dates)
        except TypeError:
            x=[x]

        y=self.int(x)
        y=y%self.period+self.offset

        if len(x)==1:
            return x[0]
        else:
            return x

class ephemeris(object):

    EPHDIR='/n/space_data/RBSP/'

    def __init__(self,scname,dstart,dend):
        self.scname=scname
        self.dstart=dstart
        self.dend=dend
        self.cached_data={}
        self.times=None
    
    def _get_variable(self,varname,f=None):
        """
        Gets the requested variable from the ephemeris file(s)
        varname: The name of the requested variable
        f (optional): A function to be applied to the variable. It should be of the form
        f(data,datetime), where data is the values of the variable, and datetime is a datetime object representing the corresponding date.
        """

        # Just return the cached copy if it's available
        try:
            return self.cached_data[varname]
        except KeyError:
            pass
        
        ordinal_dates=range(self.dstart.toordinal(),self.dend.toordinal())

        data=[]

        for ordinal in ordinal_dates:

            date=datetime.datetime.fromordinal(ordinal)
            ephfile=os.path.join(self.__class__.EPHDIR,self.scname.upper(),'MagEphem',str(date.year),'{scname:s}_def_MagEphem_T89Q_{year:d}{month:02d}{day:02d}_v1.0.0.h5'.format(scname=self.scname.lower(),year=date.year,month=date.month,day=date.day))
            ephdata=h5py.File(ephfile,'r')
            fillvalue=float(ephdata[varname].attrs['FILLVAL'][0])
            values=np.ma.masked_equal(np.array(ephdata[varname])[:-1],float(ephdata[varname].attrs['FILLVAL'][0]))
            #values=np.array(ephdata[varname])[:-1]
            #values[values==float(ephdata[varname].attrs['FILLVAL'][0])]=None
            if f:
                data.append(f(values,date))
            else:
                data.append(values)
        data=np.ma.concatenate(data)
        self.cached_data[varname]=data
        return data

    def get_variable(self,varname):
        """
        Gets the requested variable from the ephemeris file(s)
        varname: The name of the requested variable
        """

        return self._get_variable(varname)

    def get_times(self):
        """
        Get the times associated with the dates in the file
        """
        # If we have already calculated the available times, just return the array
        if self.times is not None:
            return self.times

        def utc_to_datetime(utcfloat,date):
            return date2num(date)+utcfloat/24
        self.times=self._get_variable('UTC',utc_to_datetime)
        return self.times
        
    def get_interpolator(self,varname,period=None,offset=0,jind=0,discardInvalid=True):
        """ Returns a time interpolation function for the requested variable """

        x=self.get_times()
        y=self.get_variable(varname)
        
        if len(y.shape)==2:
            y=y[:,jind]

        # Find all the valid points
        validpoints = np.where(-(x.mask + y.mask))

        y=y[validpoints]
        x=x[validpoints]

        if period:
            interpolator=DiscontinuousInterpolator(x,y,period,offset)
        else:
            interpolator=interp1d(x,y)
        return interpolator

